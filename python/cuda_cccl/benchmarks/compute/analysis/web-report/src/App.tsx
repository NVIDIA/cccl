import { useEffect, useMemo, useState } from "react";

import { AppSidebar } from "@/components/app-sidebar";
import { Badge } from "@/components/ui/badge";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type Manifest = {
  generated_at: string;
  results_base: string;
  benchmarks: BenchmarkEntry[];
};

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

type Measurement = {
  benchmark: string;
  device: number;
  axes: Record<string, string>;
  gpu_time: number;
  cpu_time: number | null;
};

type BenchmarkData = {
  py: Measurement[];
  cpp: Measurement[];
};

type AxisSelection = Record<string, string>;

type NvbenchAxisValue = {
  name?: string;
  value?: string | number;
};

type NvbenchSummaryData = {
  name?: string;
  value?: string | number;
};

type NvbenchSummary = {
  name?: string;
  data?: NvbenchSummaryData[];
};

type NvbenchState = {
  is_skipped?: boolean;
  device?: number;
  axis_values?: NvbenchAxisValue[];
  summaries?: NvbenchSummary[];
};

type NvbenchBenchmark = {
  name?: string;
  states?: NvbenchState[];
};

type NvbenchRoot = {
  benchmarks?: NvbenchBenchmark[];
};

const STANDARD_AXES = new Set(["Elements", "T", "SampleT", "OffsetT"]);
const FILTER_AXES: AxisSelection = {
  OffsetT: "I64",
};

function normalizeAxisName(name: string) {
  return name.includes("{") ? name.split("{")[0] : name;
}

function parseMeasurements(results: NvbenchRoot): Measurement[] {
  const measurements: Measurement[] = [];
  const benchmarks = results.benchmarks ?? [];

  for (const benchmark of benchmarks) {
    let benchName = String(benchmark.name ?? "unknown");
    if (benchName.startsWith("bench_")) {
      benchName = benchName.slice(6);
    }

    const states = benchmark.states ?? [];
    for (const state of states) {
      if (state.is_skipped) {
        continue;
      }

      const axes: Record<string, string> = {};
      const axisValues = state.axis_values ?? [];
      for (const axis of axisValues) {
        const axisName = normalizeAxisName(String(axis.name ?? ""));
        if (!axisName) {
          continue;
        }
        axes[axisName] = String(axis.value);
      }

      let gpuTime: number | null = null;
      let cpuTime: number | null = null;
      const summaries = state.summaries ?? [];
      for (const summary of summaries) {
        const summaryName = String(summary.name ?? "");
        const data = summary.data ?? [];

        if (
          summaryName.includes("GPU Time") &&
          !summaryName.includes("Min") &&
          !summaryName.includes("Max")
        ) {
          const value = data.find((entry) => entry.name === "value");
          if (value?.value !== undefined) {
            gpuTime = Number(value.value);
          }
        }

        if (summaryName === "CPU Time") {
          const value = data.find((entry) => entry.name === "value");
          if (value?.value !== undefined) {
            cpuTime = Number(value.value);
          }
        }
      }

      if (gpuTime !== null) {
        measurements.push({
          benchmark: benchName,
          device: Number(state.device ?? 0),
          axes,
          gpu_time: gpuTime,
          cpu_time: cpuTime,
        });
      }
    }
  }

  return measurements;
}

function formatDuration(seconds: number) {
  if (seconds >= 1) {
    return `${seconds.toFixed(3)} s`;
  }
  if (seconds >= 1e-3) {
    return `${(seconds * 1e3).toFixed(3)} ms`;
  }
  return `${(seconds * 1e6).toFixed(3)} us`;
}

function formatRatio(value: number) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return value >= 10 ? value.toFixed(2) : value.toFixed(3);
}

function formatPercentage(value: number) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function isPowerOfTwo(value: number) {
  return value > 0 && (value & (value - 1)) === 0;
}

function formatElements(value: number) {
  if (isPowerOfTwo(value)) {
    return `2^${Math.log2(value)}`;
  }
  return value.toLocaleString();
}

function filterMeasurements(
  measurements: Measurement[],
  axisFilters: AxisSelection,
  allowedAxes?: Set<string>,
) {
  let result = measurements;
  for (const [axis, preferredValue] of Object.entries(axisFilters)) {
    if (allowedAxes && !allowedAxes.has(axis)) {
      continue;
    }
    const hasAxis = result.some((measurement) => axis in measurement.axes);
    if (!hasAxis) {
      continue;
    }
    const filtered = result.filter(
      (measurement) => measurement.axes[axis] === preferredValue,
    );
    if (filtered.length) {
      result = filtered;
    }
  }
  return result;
}

function getAxisValues(measurements: Measurement[], axisName: string) {
  const values = new Set<string>();
  for (const measurement of measurements) {
    if (axisName in measurement.axes) {
      values.add(measurement.axes[axisName]);
    }
  }

  const valuesArray = [...values];
  const numericValues = valuesArray.map((value) => Number(value));
  if (numericValues.every((value) => Number.isFinite(value))) {
    return valuesArray.sort((a, b) => Number(a) - Number(b));
  }
  return valuesArray.sort();
}

function getGroupingAxes(
  measurements: Measurement[],
  allowedAxes?: Set<string>,
) {
  const axes = new Set<string>();
  for (const measurement of measurements) {
    Object.keys(measurement.axes).forEach((axis) => axes.add(axis));
  }
  return [...axes]
    .filter((axis) => !STANDARD_AXES.has(axis))
    .filter((axis) => (allowedAxes ? allowedAxes.has(axis) : true))
    .sort();
}

function average(values: number[]) {
  if (!values.length) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getTypeAxisName(measurements: Measurement[]) {
  for (const measurement of measurements) {
    if (measurement.axes.SampleT !== undefined) {
      return "SampleT";
    }
  }
  return "T";
}

function buildMeasurementMap(
  measurements: Measurement[],
  typeAxis: string,
  tValue: string | null,
) {
  const map = new Map<string, Measurement[]>();
  for (const measurement of measurements) {
    const elements = measurement.axes.Elements;
    const t = tValue ? measurement.axes[typeAxis] : "";
    if (!elements) {
      continue;
    }
    if (tValue && t !== tValue) {
      continue;
    }
    const key = `${elements}`;
    const existing = map.get(key) ?? [];
    existing.push(measurement);
    map.set(key, existing);
  }
  return map;
}

function buildTypeElementMap(measurements: Measurement[]) {
  const typeAxis = getTypeAxisName(measurements);
  const map = new Map<string, Map<number, number[]>>();
  for (const measurement of measurements) {
    const typeValue = measurement.axes[typeAxis] ?? "All";
    const elements = measurement.axes.Elements
      ? Number(measurement.axes.Elements)
      : null;
    if (elements === null || Number.isNaN(elements)) {
      continue;
    }
    const typeMap = map.get(typeValue) ?? new Map<number, number[]>();
    const values = typeMap.get(elements) ?? [];
    values.push(measurement.gpu_time);
    typeMap.set(elements, values);
    map.set(typeValue, typeMap);
  }
  return { typeAxis, map };
}

function intersect(valuesA: string[], valuesB: string[]) {
  const setB = new Set(valuesB);
  return valuesA.filter((value) => setB.has(value));
}

function resolveUrl(base: string, path: string) {
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  return new URL(path, normalizedBase).toString();
}

function getAxisSet(measurements: Measurement[]) {
  const axes = new Set<string>();
  for (const measurement of measurements) {
    Object.keys(measurement.axes).forEach((axis) => axes.add(axis));
  }
  return axes;
}

function intersectAxes(a: Set<string>, b: Set<string>) {
  const result = new Set<string>();
  for (const axis of a) {
    if (b.has(axis)) {
      result.add(axis);
    }
  }
  return result;
}

function filterByAxisSelection(
  measurements: Measurement[],
  axisValues: AxisSelection,
  allowedAxes?: Set<string>,
) {
  return measurements.filter((measurement) =>
    Object.entries(axisValues).every(
      ([axis, value]) => {
        if (allowedAxes && !allowedAxes.has(axis)) {
          return true;
        }
        if (!(axis in measurement.axes)) {
          return true;
        }
        return measurement.axes[axis] === value;
      },
    ),
  );
}

export function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [resultsBaseUrl, setResultsBaseUrl] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<BenchmarkEntry | null>(null);
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [comparisonAxisSelection, setComparisonAxisSelection] =
    useState<AxisSelection>({});
  const [rawAxisSelectionPy, setRawAxisSelectionPy] = useState<AxisSelection>(
    {},
  );
  const [rawAxisSelectionCpp, setRawAxisSelectionCpp] = useState<AxisSelection>(
    {},
  );
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const envManifest = import.meta.env.VITE_RESULTS_MANIFEST as
          | string
          | undefined;
        const candidates = [
          envManifest,
          import.meta.env.DEV ? "/results/manifest.json" : null,
          "../../results/manifest.json",
          "../results/manifest.json",
          "/results/manifest.json",
          "./manifest.json",
        ].filter(Boolean) as string[];

        let loaded = false;
        for (const candidate of candidates) {
          const response = await fetch(candidate);
          if (!response.ok) {
            continue;
          }
          const manifestData = (await response.json()) as Manifest;
          const resolvedManifestUrl =
            response.url || new URL(candidate, window.location.href).toString();
          const resolvedResultsBase = new URL(
            manifestData.results_base || ".",
            resolvedManifestUrl,
          ).toString();

          setManifest(manifestData);
          setResultsBaseUrl(resolvedResultsBase);
          setSelected(manifestData.benchmarks[0] ?? null);
          loaded = true;
          break;
        }

        if (!loaded) {
          throw new Error("manifest.json not found");
        }
      } catch (error) {
        setStatus(
          "Unable to load manifest.json. Run generate_web_report_manifest.py and serve the directory.",
        );
      }
    };
    load();
  }, []);

  useEffect(() => {
    const loadBenchmark = async () => {
      if (!manifest || !selected || !resultsBaseUrl) {
        return;
      }
      setStatus("Loading results...");
      try {
        const pyUrl = resolveUrl(resultsBaseUrl, selected.py_path);
        const cppUrl = resolveUrl(resultsBaseUrl, selected.cpp_path);
        const [pyResponse, cppResponse] = await Promise.all([
          fetch(pyUrl),
          fetch(cppUrl),
        ]);
        if (!pyResponse.ok || !cppResponse.ok) {
          throw new Error("Missing results files");
        }
        const [pyJson, cppJson] = await Promise.all([
          pyResponse.json(),
          cppResponse.json(),
        ]);
        setData({
          py: parseMeasurements(pyJson),
          cpp: parseMeasurements(cppJson),
        });
        setStatus(null);
      } catch (error) {
        setStatus("Failed to load benchmark results.");
        setData(null);
      }
    };
    loadBenchmark();
  }, [manifest, selected, resultsBaseUrl]);

  const filteredBenchmarks = useMemo(() => {
    if (!manifest) {
      return [];
    }
    const term = search.toLowerCase();
    return manifest.benchmarks.filter((entry) =>
      entry.label.toLowerCase().includes(term),
    );
  }, [manifest, search]);

  const comparisonData = useMemo(() => {
    if (!data) {
      return null;
    }
    const pyAxes = getAxisSet(data.py);
    const cppAxes = getAxisSet(data.cpp);
    const commonAxes = intersectAxes(pyAxes, cppAxes);

    return {
      commonAxes,
      py: filterMeasurements(data.py, FILTER_AXES, commonAxes),
      cpp: filterMeasurements(data.cpp, FILTER_AXES, commonAxes),
    };
  }, [data]);

  const comparisonGroupingAxes = useMemo(() => {
    if (!comparisonData) {
      return [];
    }
    return getGroupingAxes(
      [...comparisonData.py, ...comparisonData.cpp],
      comparisonData.commonAxes,
    );
  }, [comparisonData]);

  const comparisonAxisValues = useMemo(() => {
    if (!comparisonData) {
      return {} as Record<string, string[]>;
    }
    const values: Record<string, string[]> = {};
    for (const axis of comparisonGroupingAxes) {
      const pyValues = new Set(getAxisValues(comparisonData.py, axis));
      const cppValues = new Set(getAxisValues(comparisonData.cpp, axis));
      values[axis] = [...pyValues].filter((value) => cppValues.has(value));
    }
    return values;
  }, [comparisonData, comparisonGroupingAxes]);

  useEffect(() => {
    if (!comparisonGroupingAxes.length) {
      setComparisonAxisSelection({});
      return;
    }
    setComparisonAxisSelection((prev) => {
      const next: AxisSelection = { ...prev };
      for (const axis of comparisonGroupingAxes) {
        const options = comparisonAxisValues[axis] ?? [];
        if (!next[axis] || (options.length && !options.includes(next[axis]))) {
          next[axis] = options[0] ?? "";
        }
      }
      return next;
    });
  }, [comparisonGroupingAxes, comparisonAxisValues]);

  const selectedComparisonData = useMemo(() => {
    if (!comparisonData) {
      return null;
    }
    const filters: AxisSelection = { ...comparisonAxisSelection };
    return {
      py: filterByAxisSelection(comparisonData.py, filters, comparisonData.commonAxes),
      cpp: filterByAxisSelection(comparisonData.cpp, filters, comparisonData.commonAxes),
    };
  }, [comparisonData, comparisonAxisSelection]);

  const rawGroupingAxesPy = useMemo(() => {
    if (!data) {
      return [];
    }
    return getGroupingAxes(data.py);
  }, [data]);

  const rawGroupingAxesCpp = useMemo(() => {
    if (!data) {
      return [];
    }
    return getGroupingAxes(data.cpp);
  }, [data]);

  const rawAxisValuesPy = useMemo(() => {
    if (!data) {
      return {} as Record<string, string[]>;
    }
    const values: Record<string, string[]> = {};
    for (const axis of rawGroupingAxesPy) {
      values[axis] = getAxisValues(data.py, axis);
    }
    return values;
  }, [data, rawGroupingAxesPy]);

  const rawAxisValuesCpp = useMemo(() => {
    if (!data) {
      return {} as Record<string, string[]>;
    }
    const values: Record<string, string[]> = {};
    for (const axis of rawGroupingAxesCpp) {
      values[axis] = getAxisValues(data.cpp, axis);
    }
    return values;
  }, [data, rawGroupingAxesCpp]);

  useEffect(() => {
    if (!rawGroupingAxesPy.length) {
      setRawAxisSelectionPy({});
      return;
    }
    setRawAxisSelectionPy((prev) => {
      const next: AxisSelection = { ...prev };
      for (const axis of rawGroupingAxesPy) {
        const options = rawAxisValuesPy[axis] ?? [];
        if (!next[axis] || (options.length && !options.includes(next[axis]))) {
          next[axis] = options[0] ?? "";
        }
      }
      return next;
    });
  }, [rawGroupingAxesPy, rawAxisValuesPy]);

  useEffect(() => {
    if (!rawGroupingAxesCpp.length) {
      setRawAxisSelectionCpp({});
      return;
    }
    setRawAxisSelectionCpp((prev) => {
      const next: AxisSelection = { ...prev };
      for (const axis of rawGroupingAxesCpp) {
        const options = rawAxisValuesCpp[axis] ?? [];
        if (!next[axis] || (options.length && !options.includes(next[axis]))) {
          next[axis] = options[0] ?? "";
        }
      }
      return next;
    });
  }, [rawGroupingAxesCpp, rawAxisValuesCpp]);

  const tableSections = useMemo(() => {
    if (!selectedComparisonData) {
      return [];
    }
    const typeAxis = getTypeAxisName([
      ...selectedComparisonData.py,
      ...selectedComparisonData.cpp,
    ]);
    const pyTValues = getAxisValues(selectedComparisonData.py, typeAxis);
    const cppTValues = getAxisValues(selectedComparisonData.cpp, typeAxis);
    const tValues = intersect(pyTValues, cppTValues);
    const hasT = tValues.length > 0;

    const pyElements = getAxisValues(selectedComparisonData.py, "Elements");
    const cppElements = getAxisValues(selectedComparisonData.cpp, "Elements");
    const elements = intersect(pyElements, cppElements).map((value) =>
      Number(value),
    );
    elements.sort((a, b) => a - b);

    const tGroups = (hasT ? tValues : ["All"]).map((t) => {
      const pyMap = buildMeasurementMap(
        selectedComparisonData.py,
        typeAxis,
        hasT ? t : null,
      );
      const cppMap = buildMeasurementMap(
        selectedComparisonData.cpp,
        typeAxis,
        hasT ? t : null,
      );
      const rows = elements.map((size) => {
        const key = `${size}`;
        const pyValues = (pyMap.get(key) ?? []).map((item) => item.gpu_time);
        const cppValues = (cppMap.get(key) ?? []).map((item) => item.gpu_time);
        const pyAvg = average(pyValues);
        const cppAvg = average(cppValues);
        if (pyAvg === null || cppAvg === null) {
          return null;
        }
        const overhead = pyAvg - cppAvg;
        const ratio = pyAvg / cppAvg;
        const speedup = cppAvg / pyAvg;
        const pctSlower = overhead / cppAvg;
        return {
          elements: size,
          py: pyAvg,
          cpp: cppAvg,
          ratio,
          speedup,
          pctSlower,
        };
      });
      return {
        t,
        rows: rows.filter(Boolean) as NonNullable<(typeof rows)[number]>[],
      };
    });

    return tGroups;
  }, [selectedComparisonData]);

  const rawSelectedData = useMemo(() => {
    if (!data) {
      return null;
    }
    const allowedPy = new Set(rawGroupingAxesPy);
    const allowedCpp = new Set(rawGroupingAxesCpp);
    return {
      py: filterByAxisSelection(data.py, rawAxisSelectionPy, allowedPy),
      cpp: filterByAxisSelection(data.cpp, rawAxisSelectionCpp, allowedCpp),
    };
  }, [data, rawAxisSelectionPy, rawAxisSelectionCpp, rawGroupingAxesPy, rawGroupingAxesCpp]);

  const rawSections = useMemo(() => {
    if (!rawSelectedData) {
      return [] as {
        typeValue: string;
        elements: number[];
        cpp?: Map<number, number>;
        py?: Map<number, number>;
      }[];
    }

    const cppData = buildTypeElementMap(rawSelectedData.cpp);
    const pyData = buildTypeElementMap(rawSelectedData.py);

    const typeValues = new Set<string>();
    for (const key of cppData.map.keys()) {
      typeValues.add(key);
    }
    for (const key of pyData.map.keys()) {
      typeValues.add(key);
    }

    const sections: {
      typeValue: string;
      elements: number[];
      cpp?: Map<number, number>;
      py?: Map<number, number>;
    }[] = [];

    for (const typeValue of Array.from(typeValues).sort()) {
      const cppMap = cppData.map.get(typeValue) ?? new Map();
      const pyMap = pyData.map.get(typeValue) ?? new Map();

      const elementSet = new Set<number>();
      for (const key of cppMap.keys()) {
        elementSet.add(key);
      }
      for (const key of pyMap.keys()) {
        elementSet.add(key);
      }

      const elements = Array.from(elementSet).sort((a, b) => a - b);

      const cppAverages = new Map<number, number>();
      for (const [key, values] of cppMap.entries()) {
        const avg = average(values);
        if (avg !== null) {
          cppAverages.set(key, avg);
        }
      }
      const pyAverages = new Map<number, number>();
      for (const [key, values] of pyMap.entries()) {
        const avg = average(values);
        if (avg !== null) {
          pyAverages.set(key, avg);
        }
      }

      sections.push({
        typeValue,
        elements,
        cpp: cppAverages,
        py: pyAverages,
      });
    }

    return sections;
  }, [rawSelectedData]);

  return (
    <SidebarProvider>
      <AppSidebar
        benchmarks={filteredBenchmarks}
        selectedId={selected?.id ?? null}
        search={search}
        onSearch={setSearch}
        onSelectBenchmark={(entry) => setSelected(entry)}
      />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b bg-background px-4">
          <SidebarTrigger className="-ml-1" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbPage>
                  {selected?.label ?? "Loading benchmarks"}
                </BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <div className="ml-auto hidden items-center gap-2 text-xs text-muted-foreground md:flex">
            {selected?.device?.name ? (
              <Badge variant="secondary">{selected.device.name}</Badge>
            ) : null}
            {manifest?.generated_at ? (
              <Badge variant="secondary">
                Generated {new Date(manifest.generated_at).toLocaleString()}
              </Badge>
            ) : null}
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-6 p-4">
          <Card className="overflow-hidden rounded-none border-0 shadow-none">
            <CardHeader className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-lg">C++ vs Python</CardTitle>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6 pt-6">
              {comparisonGroupingAxes.length > 0 && (
                <div className="flex flex-wrap items-end gap-3">
                  {comparisonGroupingAxes.map((axis) => (
                    <div key={axis} className="min-w-[160px] space-y-1">
                      <div className="text-xs font-medium text-muted-foreground">
                        {axis}
                      </div>
                        <Select
                          value={comparisonAxisSelection[axis] ?? ""}
                          onValueChange={(value) =>
                            setComparisonAxisSelection((prev) => ({
                              ...prev,
                              [axis]: value ?? "",
                            }))
                          }
                        >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder={`Select ${axis}`} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectGroup>
                            <SelectLabel>{axis}</SelectLabel>
                            {comparisonAxisValues[axis]?.map((value) => (
                              <SelectItem key={value} value={value}>
                                {value}
                              </SelectItem>
                            ))}
                          </SelectGroup>
                        </SelectContent>
                      </Select>
                    </div>
                  ))}
                </div>
              )}
              {status && (
                <div className="rounded-lg border border-dashed border-border bg-muted/30 p-4 text-sm text-muted-foreground">
                  {status}
                </div>
              )}
              {!status && tableSections.length === 0 && (
                <div className="rounded-lg border border-dashed border-border bg-muted/30 p-4 text-sm text-muted-foreground">
                  No overlapping measurements found for the current filters.
                </div>
              )}
              {tableSections.map((section) => (
                <div key={section.t} className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-base font-semibold">
                      {section.t === "All" ? "All types" : `T = ${section.t}`}
                    </h3>
                  </div>

                  <div className="overflow-hidden rounded-xl border border-border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[160px]">Elements</TableHead>
                          <TableHead className="text-right">C++ GPU</TableHead>
                          <TableHead className="text-right">
                            Python GPU
                          </TableHead>
                          <TableHead className="text-right">Speedup</TableHead>
                          <TableHead className="text-right">% Slower</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {section.rows.map((row) => {
                          const isSlow = row.pctSlower > 0.1;
                          return (
                            <TableRow
                              key={row.elements}
                              className={
                                isSlow
                                  ? "bg-destructive/10 text-destructive"
                                  : undefined
                              }
                            >
                              <TableCell className="font-medium">
                                {formatElements(row.elements)}
                              </TableCell>
                              <TableCell className="text-right">
                                {formatDuration(row.cpp)}
                              </TableCell>
                              <TableCell className="text-right">
                                {formatDuration(row.py)}
                              </TableCell>
                              <TableCell className="text-right">
                                {formatRatio(row.speedup)}x
                              </TableCell>
                              <TableCell className="text-right">
                                {formatPercentage(row.pctSlower)}
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
          <Card className="overflow-hidden rounded-none border-0 shadow-none">
            <CardHeader className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-lg">Raw results</CardTitle>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-6 pt-6">
              <div className="grid gap-6 lg:grid-cols-2">
                <div className="space-y-4">
                  <div>
                    <div className="text-base font-semibold">C++</div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {rawGroupingAxesCpp.length === 0 && (
                      <div className="rounded-lg border border-dashed border-border bg-muted/40 p-3 text-sm text-muted-foreground sm:col-span-2">
                        No additional axes detected for this benchmark.
                      </div>
                    )}
                    {rawGroupingAxesCpp.map((axis) => (
                      <div key={axis} className="space-y-1">
                        <div className="text-xs font-medium text-muted-foreground">
                          {axis}
                        </div>
                        <Select
                          value={rawAxisSelectionCpp[axis] ?? ""}
                          onValueChange={(value) =>
                            setRawAxisSelectionCpp((prev) => ({
                              ...prev,
                              [axis]: value ?? "",
                            }))
                          }
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder={`Select ${axis}`} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectLabel>{axis}</SelectLabel>
                              {rawAxisValuesCpp[axis]?.map((value) => (
                                <SelectItem key={value} value={value}>
                                  {value}
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="text-base font-semibold">Python</div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {rawGroupingAxesPy.length === 0 && (
                      <div className="rounded-lg border border-dashed border-border bg-muted/40 p-3 text-sm text-muted-foreground sm:col-span-2">
                        No additional axes detected for this benchmark.
                      </div>
                    )}
                    {rawGroupingAxesPy.map((axis) => (
                      <div key={axis} className="space-y-1">
                        <div className="text-xs font-medium text-muted-foreground">
                          {axis}
                        </div>
                        <Select
                          value={rawAxisSelectionPy[axis] ?? ""}
                          onValueChange={(value) =>
                            setRawAxisSelectionPy((prev) => ({
                              ...prev,
                              [axis]: value ?? "",
                            }))
                          }
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder={`Select ${axis}`} />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectLabel>{axis}</SelectLabel>
                              {rawAxisValuesPy[axis]?.map((value) => (
                                <SelectItem key={value} value={value}>
                                  {value}
                                </SelectItem>
                              ))}
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="space-y-6">
                {rawSections.map((section) => (
                  <div key={section.typeValue} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-base font-semibold">
                        {section.typeValue === "All"
                          ? "All types"
                          : `T = ${section.typeValue}`}
                      </h3>
                    </div>
                    <div className="grid gap-4 lg:grid-cols-2">
                      <div className="space-y-2">
                        <div className="overflow-hidden rounded-xl border border-border">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="w-[160px]">
                                  Elements
                                </TableHead>
                                <TableHead className="text-right">
                                  GPU Time
                                </TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {section.elements.map((elements) => (
                                <TableRow key={`cpp-${elements}`}>
                                  <TableCell className="font-medium">
                                    {formatElements(elements)}
                                  </TableCell>
                                  <TableCell className="text-right">
                                    {section.cpp?.has(elements)
                                      ? formatDuration(
                                        section.cpp.get(elements) ?? 0,
                                      )
                                      : "-"}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="overflow-hidden rounded-xl border border-border">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="w-[160px]">
                                  Elements
                                </TableHead>
                                <TableHead className="text-right">
                                  GPU Time
                                </TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {section.elements.map((elements) => (
                                <TableRow key={`py-${elements}`}>
                                  <TableCell className="font-medium">
                                    {formatElements(elements)}
                                  </TableCell>
                                  <TableCell className="text-right">
                                    {section.py?.has(elements)
                                      ? formatDuration(
                                        section.py.get(elements) ?? 0,
                                      )
                                      : "-"}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          <footer className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
            <div className="flex flex-wrap items-center gap-3" />
          </footer>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}

export default App;
