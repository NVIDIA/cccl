# Independent formal follow-up

Latency columns are medians of all process medians (µs for consumers, ms for models).

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| integer-65536 | first-thread/32 | 8 | 46.063 | 46.071 | 0.9999× | 0.9992–1.0007 | within default bound |
| integer-65536 | first-thread/128 | 8 | 41.570 | 41.579 | 0.9991× | 0.9979–1.0002 | within default bound |
| integer-65536 | first-thread/256 | 8 | 88.667 | 88.666 | 1.0000× | 0.9999–1.0000 | within default bound |
| integer-65536 | first-thread/512 | 8 | 187.919 | 187.930 | 0.9999× | 0.9998–1.0001 | within default bound |
| integer-65536 | normalization/32 | 8 | 47.200 | 47.384 | 0.9973× | 0.9962–0.9984 | no stable >1% gain |
| integer-65536 | normalization/128 | 8 | 68.617 | 65.306 | 1.0503× | 1.0494–1.0511 | stable >1% gain |
| integer-65536 | normalization/256 | 8 | 135.516 | 132.288 | 1.0247× | 1.0243–1.0250 | stable >1% gain |
| integer-65536 | normalization/512 | 8 | 284.124 | 273.994 | 1.0371× | 1.0369–1.0374 | stable >1% gain |
| retained-65536 | first-thread/32 | 8 | 50.253 | 50.256 | 0.9999× | 0.9986–1.0013 | within default bound |
| retained-65536 | first-thread/128 | 8 | 45.550 | 45.556 | 1.0000× | 0.9987–1.0013 | within default bound |
| retained-65536 | first-thread/256 | 8 | 108.429 | 108.452 | 1.0000× | 0.9993–1.0006 | within default bound |
| retained-65536 | first-thread/512 | 8 | 233.140 | 233.148 | 0.9999× | 0.9997–1.0001 | within default bound |
| retained-65536 | normalization/32 | 8 | 46.146 | 45.891 | 1.0066× | 1.0048–1.0084 | no stable >1% gain |
| retained-65536 | normalization/128 | 8 | 66.683 | 64.442 | 1.0350× | 1.0338–1.0361 | stable >1% gain |
| retained-65536 | normalization/256 | 8 | 142.292 | 141.351 | 1.0066× | 1.0059–1.0073 | no stable >1% gain |
| retained-65536 | normalization/512 | 8 | 306.627 | 303.690 | 1.0096× | 1.0092–1.0100 | no stable >1% gain |
