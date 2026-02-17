import * as React from "react";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

type BenchmarkEntry = {
  id: string;
  label: string;
  category: string;
  name: string;
  py_path: string;
  cpp_path: string;
  axes: string[];
  device?: {
    id: number;
    name: string;
    sm_version?: number;
  };
};

type AppSidebarProps = Omit<React.ComponentProps<typeof Sidebar>, "onSelect"> & {
  benchmarks: BenchmarkEntry[];
  selectedId: string | null;
  search: string;
  onSearch: (value: string) => void;
  onSelectBenchmark: (entry: BenchmarkEntry) => void;
};

function groupByCategory(benchmarks: BenchmarkEntry[]) {
  const groups = new Map<string, BenchmarkEntry[]>();
  for (const entry of benchmarks) {
    const key = entry.category || "Other";
    const items = groups.get(key) ?? [];
    items.push(entry);
    groups.set(key, items);
  }
  return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

export function AppSidebar({
  benchmarks,
  selectedId,
  search,
  onSearch,
  onSelectBenchmark,
  ...props
}: AppSidebarProps) {
  const grouped = groupByCategory(benchmarks);

  return (
    <Sidebar variant="inset" collapsible="offcanvas" {...props}>
      <SidebarHeader className="gap-3 border-b px-3 py-4">
        <div>
          <div className="text-sm font-semibold">cuda.compute benchmarks</div>
        </div>
        <SidebarInput
          placeholder="Search benchmarks"
          value={search}
          onChange={(event) => onSearch(event.target.value)}
        />
      </SidebarHeader>
      <SidebarContent className="pb-4">
        {grouped.map(([category, items]) => (
          <SidebarGroup key={category}>
            <SidebarGroupLabel>{category}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {items.map((entry) => (
                  <SidebarMenuItem key={entry.id}>
                    <SidebarMenuButton
                      onClick={() => onSelectBenchmark(entry)}
                      isActive={selectedId === entry.id}
                    >
                      <span>{entry.name}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
        {!benchmarks.length && (
          <div className="px-3 text-xs text-muted-foreground">
            No benchmarks match that filter.
          </div>
        )}
      </SidebarContent>
    </Sidebar>
  );
}
