// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <trait_catalog.h>

#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <common/type_traits.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

namespace
{
enum class OutputMode : std::uint8_t
{
  CHECK,
  OUTPUT,
};

class ParsedArgs
{
public:
  struct Options
  {
    llvm::cl::OptionCategory category{"CCCL trait catalog generation options"};

    llvm::cl::opt<OutputMode> mode{
      "mode",
      llvm::cl::desc{"Set tool mode"},
      llvm::cl::values(
        clEnumValN(
          OutputMode::CHECK, "check", "Check the output against existing catalog and return nonzero if different"),
        clEnumValN(OutputMode::OUTPUT, "output", "Output the updated trait catalog")),
      llvm::cl::cat{category}};
    llvm::cl::opt<std::string> output{
      "output",
      llvm::cl::ValueOptional,
      llvm::cl::init("-"),
      llvm::cl::desc{"File to emit catalog to if --mode=output (defaults to stdout). Passing this flag implies "
                     "--mode=output"},
      llvm::cl::cat{category},
      llvm::cl::callback([&](const std::string& value) {
        mode = OutputMode::OUTPUT;
        if (value.empty())
        {
          // Note, don't use = for assignment, as it calls this callback again
          output.setValue("-"); // NOLINT(bugprone-argument-comment)
        }
      })};
    llvm::cl::list<std::string> toplevel_namespaces{
      "namespaces",
      llvm::cl::desc{"List of namespaces to scan for _t and _v template candidates. Namespaces must be fully qualified "
                     "(e.g. \"cuda::std\" or \"cub::nested_namespace\"), but must NOT contain a leading or trailing "
                     "\"::\" (since those match against nothing)."},
      llvm::cl::OneOrMore,
      llvm::cl::CommaSeparated,
      llvm::cl::ValueRequired,
      llvm::cl::cat{category}};
  };

  Options options;
  clang::tooling::CommonOptionsParser parser;

  [[nodiscard]] static llvm::Expected<std::unique_ptr<ParsedArgs>> create(int argc, const char* argv[])
  {
    constexpr auto err_to_expected = [](auto& option, llvm::Twine msg) {
      std::string message;
      llvm::raw_string_ostream stream{message};

      option.error(msg, stream);
      return llvm::createStringError(message);
    };

    // Can't use make_unique since the ctor is private
    auto ret = std::unique_ptr<ParsedArgs>(new ParsedArgs{argc, argv});

    for (llvm::StringRef ns : ret->options.toplevel_namespaces)
    {
      if (ns.starts_with("::"))
      {
        return err_to_expected(ret->options.toplevel_namespaces,
                               "invalid namespace '" + ns + "', must not begin with \"::\"");
      }
      if (ns.ends_with("::"))
      {
        return err_to_expected(ret->options.toplevel_namespaces,
                               "invalid namespace '" + ns + "', must not end with \"::\"");
      }
    }
    return ret;
  }

private:
  explicit ParsedArgs(int argc, const char* argv[])
      : parser{llvm::ExitOnError{}(clang::tooling::CommonOptionsParser::create(argc, argv, options.category))}
  {}
};

class TraitCollector final : public clang::ast_matchers::MatchFinder::MatchCallback
{
  llvm::SmallVector<std::string> target_nses_;

public:
  template <class Range>
  explicit TraitCollector(const Range& range)
      : target_nses_{[&] {
        const auto dedup = llvm::StringSet<>{llvm::from_range, range};

        return llvm::to_vector(llvm::map_range(dedup, [](const auto& entry) {
          return entry.getKey().str();
        }));
      }()}
  {}

  // NOLINTBEGIN(cppcoreguidelines-non-private-member-variables-in-classes)
  llvm::StringSet<> bases;
  llvm::StringSet<> types;
  llvm::StringSet<> values;
  // NOLINTEND(cppcoreguidelines-non-private-member-variables-in-classes)

  static constexpr llvm::StringRef DECL_BIND{""};

  void run(const clang::ast_matchers::MatchFinder::MatchResult& result) override
  {
    const auto* declaration = result.Nodes.getNodeAs<clang::NamedDecl>(DECL_BIND);

    if (const auto* variable = llvm::dyn_cast_or_null<clang::VarDecl>(declaration))
    {
      declaration = variable->getDescribedVarTemplate();
    }

    if (!declaration || !declaration->getDeclName().isIdentifier())
    {
      return;
    }

    const auto name     = clang::tidy::cccl::util::fully_qualified_name(*declaration);
    const auto name_ref = llvm::StringRef{name};

    if (!llvm::any_of(target_nses_, [&](llvm::StringRef ns) {
          auto suffix = name_ref;

          return suffix.consume_front(ns) && suffix.starts_with("::");
        }))
    {
      return;
    }

    if (llvm::isa<clang::ClassTemplateDecl, clang::TypeAliasTemplateDecl>(declaration))
    {
      bases.insert(name);
      if (llvm::isa<clang::TypeAliasTemplateDecl>(declaration) && declaration->getName().ends_with("_t"))
      {
        types.insert(name_ref.drop_back(2));
      }
    }

    if (llvm::isa<clang::VarTemplateDecl>(declaration) && declaration->getName().ends_with("_v"))
    {
      values.insert(name_ref.drop_back(2));
    }
  }
};

[[nodiscard]] llvm::Expected<TraitCollector> gather_traits(const std::unique_ptr<ParsedArgs>& args)
{
  auto ret = TraitCollector{args->options.toplevel_namespaces};

  using namespace clang::ast_matchers;

  auto finder = MatchFinder{};

  finder.addMatcher(
    namedDecl(anyOf(classTemplateDecl(), typeAliasTemplateDecl(), varDecl()), hasDeclContext(namespaceDecl()))
      .bind(TraitCollector::DECL_BIND),
    &ret);

  auto tool = clang::tooling::ClangTool{args->parser.getCompilations(), args->parser.getSourcePathList()};
  // Need to parse -resource-dir from the compile_commands output for the file, otherwise clang
  // tooling cannot find builtin headers like stdarg.h
  tool.appendArgumentsAdjuster([](const clang::tooling::CommandLineArguments& args, llvm::StringRef file) {
    return clang::tooling::getInsertArgumentAdjuster(("-resource-dir=" + clang::GetResourcesPath(args.front())).c_str(),
                                                     clang::tooling::ArgumentInsertPosition::BEGIN)(args, file);
  });

  auto fact = clang::tooling::newFrontendActionFactory(&finder);

  if (const auto status = tool.run(fact.get()); status != 0)
  {
    return llvm::createStringError(/*Fmt=*/"ClangTool failed with exit status %d", status);
  }

  return ret;
}

struct UpdatedCatalog
{
  llvm::StringSet<> types;
  llvm::StringSet<> values;
  bool missing;
};

[[nodiscard]] llvm::Expected<UpdatedCatalog>
update_catalog(const std::unique_ptr<ParsedArgs>& args, const TraitCollector& discovered)
{
  auto ret = UpdatedCatalog{};

  // Take by copy so we can update them
  ret.types   = clang::tidy::cccl::TYPE_TRAITS();
  ret.values  = clang::tidy::cccl::VALUE_TRAITS();
  ret.missing = false;

  const auto warn  = args->options.mode == OutputMode::CHECK;
  const auto merge = [&](const llvm::StringSet<>& candidates, llvm::StringSet<>& catalog, llvm::StringRef name) {
    for (const auto& entry : candidates)
    {
      // Keep entries from configurations that these translation units cannot expose.
      if (!discovered.bases.contains(entry.getKey()) || !catalog.insert(entry.getKey()).second)
      {
        continue;
      }

      ret.missing = true;

      if (warn)
      {
        llvm::errs() << name << " is missing " << entry.getKey() << '\n';
      }
    }
  };

  merge(discovered.types, ret.types, "TYPE_TRAITS");
  merge(discovered.values, ret.values, "VALUE_TRAITS");
  return ret;
}

void emit_set(llvm::raw_ostream& stream, llvm::StringRef name, const llvm::StringSet<>& traits)
{
  std::vector<llvm::StringRef> names;

  names.reserve(traits.size());
  for (const auto& entry : traits)
  {
    names.push_back(entry.getKey());
  }

  std::sort(names.begin(), names.end());

  stream << "[[nodiscard]] inline const llvm::StringSet<>& " << name << "()\n";
  stream << "{\n";
  stream << "  static const auto values = llvm::StringSet<>{\n";
  for (auto&& entry : names)
  {
    stream << "    \"" << entry << "\",\n";
  }
  stream << "  };\n";
  stream << '\n';
  stream << "  return values;\n";
  stream << "}\n";
  stream << '\n';
}

[[nodiscard]] llvm::Error write_output(const std::unique_ptr<ParsedArgs>& args, const UpdatedCatalog& catalog)
{
  auto cb = [&](llvm::raw_ostream& stream) {
    stream << R"cpp(// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generated by generate_trait_catalog.cpp, do not edit.
//
// Note: entries absent from the scanned configurations are preserved because it is too hard to
// determine whether they actually disappeared or whether they are not hidden behind a
// configuration we can't see.

#pragma once

#include <llvm/ADT/StringSet.h>

namespace clang::tidy::cccl
{
)cpp";

    emit_set(stream, "TYPE_TRAITS", catalog.types);
    emit_set(stream, "VALUE_TRAITS", catalog.values);

    stream << "} // namespace clang::tidy::cccl\n";

    return llvm::Error::success();
  };

  return llvm::writeToOutput(args->options.output, std::move(cb));
}
} // namespace

int main(int argc, const char* argv[])
{
  llvm::ExitOnError check;

  const auto args       = check(ParsedArgs::create(argc, argv));
  const auto discovered = check(gather_traits(args));
  const auto catalog    = check(update_catalog(args, discovered));

  switch (args->options.mode.getValue())
  {
    case OutputMode::CHECK:
      if (catalog.missing)
      {
        llvm::errs() << '\n';
        llvm::errs()
          << "Run to update inplace: " //
          << llvm::join(argv, argv + argc, " ") //
          << " --output=" << llvm::sys::path::parent_path(__FILE__).str() << "/trait_catalog.h" << '\n';
        return EXIT_FAILURE;
      }
      break;
    case OutputMode::OUTPUT:
      check(write_output(args, catalog));
      break;
  }
  return EXIT_SUCCESS;
}
