# Independent formal follow-up

Latency columns are medians of all process medians (µs for consumers, ms for models).

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| integer-8192 | first-thread/32 | 8 | 22.582 | 22.604 | 0.9998× | 0.9993–1.0004 | within default bound |
| integer-8192 | first-thread/128 | 8 | 22.683 | 22.645 | 1.0017× | 0.9990–1.0044 | within default bound |
| integer-8192 | first-thread/256 | 8 | 42.181 | 42.198 | 0.9996× | 0.9988–1.0004 | within default bound |
| integer-8192 | first-thread/512 | 8 | 93.646 | 93.512 | 1.0004× | 0.9995–1.0012 | within default bound |
| integer-8192 | normalization/32 | 8 | 26.711 | 26.755 | 0.9988× | 0.9983–0.9994 | no stable >1% gain |
| integer-8192 | normalization/128 | 8 | 37.982 | 35.799 | 1.0613× | 1.0587–1.0636 | stable >1% gain |
| integer-8192 | normalization/256 | 8 | 75.891 | 72.168 | 1.0524× | 1.0518–1.0531 | stable >1% gain |
| integer-8192 | normalization/512 | 8 | 162.258 | 154.253 | 1.0522× | 1.0519–1.0525 | stable >1% gain |
| legacy-8192 | first-thread/32 | 8 | 22.602 | 22.586 | 1.0006× | 0.9996–1.0016 | within default bound |
| legacy-8192 | first-thread/128 | 8 | 28.890 | 28.889 | 0.9991× | 0.9979–1.0002 | within default bound |
| legacy-8192 | first-thread/256 | 8 | 59.475 | 59.464 | 1.0002× | 0.9999–1.0006 | within default bound |
| legacy-8192 | first-thread/512 | 8 | 120.340 | 120.363 | 0.9998× | 0.9989–1.0005 | within default bound |
| legacy-8192 | normalization/32 | 8 | 27.099 | 26.839 | 1.0102× | 1.0076–1.0129 | no stable >1% gain |
| legacy-8192 | normalization/128 | 8 | 39.049 | 36.023 | 1.0839× | 1.0827–1.0852 | stable >1% gain |
| legacy-8192 | normalization/256 | 8 | 77.913 | 77.919 | 1.0000× | 0.9998–1.0002 | no stable >1% gain |
| legacy-8192 | normalization/512 | 8 | 166.116 | 166.074 | 1.0000× | 0.9993–1.0007 | no stable >1% gain |
| retained-8192 | first-thread/32 | 8 | 22.596 | 22.606 | 0.9999× | 0.9988–1.0011 | within default bound |
| retained-8192 | first-thread/128 | 8 | 28.744 | 28.764 | 0.9988× | 0.9974–1.0001 | within default bound |
| retained-8192 | first-thread/256 | 8 | 59.481 | 59.470 | 1.0001× | 0.9998–1.0004 | within default bound |
| retained-8192 | first-thread/512 | 8 | 120.414 | 120.239 | 1.0007× | 0.9996–1.0018 | within default bound |
| retained-8192 | normalization/32 | 8 | 27.121 | 26.866 | 1.0080× | 1.0052–1.0110 | no stable >1% gain |
| retained-8192 | normalization/128 | 8 | 39.112 | 36.099 | 1.0830× | 1.0813–1.0847 | stable >1% gain |
| retained-8192 | normalization/256 | 8 | 77.924 | 77.921 | 1.0000× | 0.9998–1.0001 | no stable >1% gain |
| retained-8192 | normalization/512 | 8 | 166.120 | 166.056 | 1.0000× | 0.9992–1.0008 | no stable >1% gain |
