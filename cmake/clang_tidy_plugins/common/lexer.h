// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <utility>

#include <clang-tidy/utils/LexerUtils.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/TokenKinds.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/Token.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Config/llvm-config.h>

namespace clang::tidy::cccl::util
{
#if LLVM_VERSION_MAJOR >= 23
[[nodiscard]] inline CharSourceRange findTokenTextInRange(
  CharSourceRange Range,
  const SourceManager& SM,
  const LangOptions& LangOpts,
  llvm::function_ref<bool(const Token&)> Pred)
{
  return utils::lexer::findTokenTextInRange(std::move(Range), SM, LangOpts, std::move(Pred));
}
#else
// Backport of https://github.com/llvm/llvm-project/blob/main/clang-tools-extra/clang-tidy/utils/LexerUtils.cpp.
[[nodiscard]] inline CharSourceRange findTokenTextInRange(
  CharSourceRange Range,
  const SourceManager& SM,
  const LangOptions& LangOpts,
  llvm::function_ref<bool(const Token&)> Pred)
{
  if (Range.isInvalid())
  {
    return {};
  }

  // Normalize to a file-based char range so raw lexing can operate on one
  // contiguous buffer and reject unmappable (e.g. macro) ranges.
  const CharSourceRange FileRange = Lexer::makeFileCharRange(Range, SM, LangOpts);

  if (FileRange.isInvalid())
  {
    return {};
  }

  const auto [BeginFID, BeginOffset] = SM.getDecomposedLoc(FileRange.getBegin());
  const auto [EndFID, EndOffset]     = SM.getDecomposedLoc(FileRange.getEnd());

  if (BeginFID != EndFID || BeginOffset > EndOffset)
  {
    return {};
  }

  bool Invalid           = false;
  const StringRef Buffer = SM.getBufferData(BeginFID, &Invalid);
  if (Invalid)
  {
    return {};
  }

  const char* LexStart = Buffer.data() + BeginOffset;
  // Re-lex raw tokens in the bounded file buffer while preserving comments so
  // callers can match tokens regardless of interleaved comments.
  Lexer TheLexer{SM.getLocForStartOfFile(BeginFID), LangOpts, Buffer.begin(), LexStart, Buffer.end()};
  TheLexer.SetCommentRetentionState(true);

  while (true)
  {
    Token Tok{};
    if (TheLexer.LexFromRawLexer(Tok))
    {
      return {};
    }

    if (Tok.is(tok::eof) || Tok.getLocation() == FileRange.getEnd()
        || SM.isBeforeInTranslationUnit(FileRange.getEnd(), Tok.getLocation()))
    {
      return {};
    }

    if (!Pred(Tok))
    {
      continue;
    }

    Token NextTok{};
    if (TheLexer.LexFromRawLexer(NextTok))
    {
      return {};
    }
    // Return a char range ending at the next token start so trailing trivia of
    // the matched token is included (useful for fix-it removals).
    return CharSourceRange::getCharRange(Tok.getLocation(), NextTok.getLocation());
  }
}
#endif
} // namespace clang::tidy::cccl::util
