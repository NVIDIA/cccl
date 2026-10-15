# Independent formal follow-up

Latency columns are medians of all process medians (µs for consumers, ms for models).

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| model-torch-order-b1 | model/128 | 8 | 462.562 | 465.099 | 0.9864× | 0.9670–1.0036 | no stable >1% gain |
| model-torch-order-b1 | model/512 | 8 | 466.214 | 469.129 | 1.0025× | 0.9802–1.0278 | no stable >1% gain |
| model-torch-order-b16 | model/128 | 8 | 499.997 | 497.443 | 1.0057× | 0.9948–1.0178 | no stable >1% gain |
| model-torch-order-b16 | model/512 | 8 | 766.965 | 767.565 | 0.9993× | 0.9940–1.0049 | no stable >1% gain |
