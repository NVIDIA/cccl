# Independent formal follow-up

Latency columns are medians of all process medians (µs for consumers, ms for models).

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| integer-8192 | first-thread/32 | 8 | 7.199 | 7.149 | 1.0040× | 1.0005–1.0074 | within default bound |
| integer-8192 | first-thread/128 | 8 | 7.019 | 6.961 | 1.0050× | 1.0021–1.0081 | within default bound |
| integer-8192 | first-thread/256 | 8 | 10.565 | 10.514 | 1.0025× | 1.0004–1.0046 | within default bound |
| integer-8192 | first-thread/512 | 8 | 20.042 | 19.998 | 1.0011× | 0.9999–1.0024 | within default bound |
| integer-8192 | normalization/32 | 8 | 7.355 | 7.302 | 1.0012× | 0.9982–1.0046 | no stable >1% gain |
| integer-8192 | normalization/128 | 8 | 9.052 | 8.478 | 1.0640× | 1.0615–1.0669 | stable >1% gain |
| integer-8192 | normalization/256 | 8 | 15.829 | 15.247 | 1.0375× | 1.0357–1.0395 | stable >1% gain |
| integer-8192 | normalization/512 | 8 | 31.316 | 29.759 | 1.0523× | 1.0515–1.0530 | stable >1% gain |
| retained-8192 | first-thread/32 | 8 | 7.498 | 7.480 | 0.9993× | 0.9934–1.0051 | within default bound |
| retained-8192 | first-thread/128 | 8 | 7.474 | 7.480 | 0.9986× | 0.9927–1.0042 | within default bound |
| retained-8192 | first-thread/256 | 8 | 13.038 | 13.039 | 1.0004× | 0.9972–1.0036 | within default bound |
| retained-8192 | first-thread/512 | 8 | 25.684 | 25.677 | 1.0007× | 0.9995–1.0019 | within default bound |
| retained-8192 | normalization/32 | 8 | 7.140 | 7.081 | 1.0091× | 1.0032–1.0150 | no stable >1% gain |
| retained-8192 | normalization/128 | 8 | 9.042 | 8.658 | 1.0438× | 1.0394–1.0486 | stable >1% gain |
| retained-8192 | normalization/256 | 8 | 16.714 | 16.497 | 1.0137× | 1.0112–1.0164 | stable >1% gain |
| retained-8192 | normalization/512 | 8 | 34.926 | 34.388 | 1.0161× | 1.0147–1.0175 | stable >1% gain |
