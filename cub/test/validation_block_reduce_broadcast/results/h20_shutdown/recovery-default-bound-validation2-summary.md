# Independent formal follow-up

Latency columns are medians of all process medians (µs for consumers, ms for models).

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| legacy-8192-validation2 | first-thread/32 | 8 | 7.309 | 7.409 | 0.9938× | 0.9885–0.9992 | default bound not met |
| legacy-8192-validation2 | first-thread/128 | 8 | 7.425 | 7.479 | 0.9966× | 0.9913–1.0026 | within default bound |
| legacy-8192-validation2 | first-thread/256 | 8 | 12.944 | 13.023 | 0.9959× | 0.9932–0.9987 | within default bound |
| legacy-8192-validation2 | first-thread/512 | 8 | 25.553 | 25.650 | 0.9984× | 0.9969–1.0000 | within default bound |
| legacy-8192-validation2 | normalization/32 | 8 | 7.272 | 7.291 | 1.0004× | 0.9947–1.0077 | no stable >1% gain |
| legacy-8192-validation2 | normalization/128 | 8 | 14.269 | 13.951 | 1.0250× | 1.0224–1.0279 | stable >1% gain |
| legacy-8192-validation2 | normalization/256 | 8 | 26.641 | 26.484 | 1.0079× | 1.0064–1.0097 | no stable >1% gain |
| legacy-8192-validation2 | normalization/512 | 8 | 53.729 | 53.244 | 1.0094× | 1.0086–1.0103 | no stable >1% gain |
