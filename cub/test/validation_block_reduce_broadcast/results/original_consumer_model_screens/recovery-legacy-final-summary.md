# Independent formal follow-up

Latency columns are medians of all process medians (µs for consumers, ms for models).

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| legacy-65536 | first-thread/32 | 8 | 50.152 | 50.205 | 0.9995× | 0.9989–1.0001 | within default bound |
| legacy-65536 | first-thread/128 | 8 | 45.427 | 45.491 | 0.9994× | 0.9985–1.0001 | within default bound |
| legacy-65536 | first-thread/256 | 8 | 108.352 | 108.404 | 0.9998× | 0.9993–1.0002 | within default bound |
| legacy-65536 | first-thread/512 | 8 | 233.154 | 233.154 | 1.0000× | 0.9999–1.0001 | within default bound |
| legacy-65536 | normalization/32 | 8 | 44.745 | 44.655 | 1.0025× | 1.0015–1.0033 | no stable >1% gain |
| legacy-65536 | normalization/128 | 8 | 102.122 | 99.620 | 1.0251× | 1.0245–1.0256 | stable >1% gain |
| legacy-65536 | normalization/256 | 8 | 204.388 | 202.653 | 1.0088× | 1.0085–1.0091 | no stable >1% gain |
| legacy-65536 | normalization/512 | 8 | 429.658 | 427.362 | 1.0052× | 1.0051–1.0053 | no stable >1% gain |
| legacy-8192 | first-thread/32 | 8 | 7.324 | 7.413 | 0.9970× | 0.9897–1.0051 | default bound not met |
| legacy-8192 | first-thread/128 | 8 | 7.432 | 7.486 | 0.9981× | 0.9910–1.0059 | within default bound |
| legacy-8192 | first-thread/256 | 8 | 12.950 | 13.032 | 0.9974× | 0.9932–1.0018 | within default bound |
| legacy-8192 | first-thread/512 | 8 | 25.547 | 25.650 | 0.9987× | 0.9962–1.0014 | within default bound |
| legacy-8192 | normalization/32 | 8 | 7.284 | 7.291 | 1.0019× | 0.9951–1.0091 | no stable >1% gain |
| legacy-8192 | normalization/128 | 8 | 14.273 | 13.927 | 1.0271× | 1.0233–1.0314 | stable >1% gain |
| legacy-8192 | normalization/256 | 8 | 26.667 | 26.477 | 1.0089× | 1.0068–1.0113 | no stable >1% gain |
| legacy-8192 | normalization/512 | 8 | 53.682 | 53.242 | 1.0095× | 1.0082–1.0107 | no stable >1% gain |
