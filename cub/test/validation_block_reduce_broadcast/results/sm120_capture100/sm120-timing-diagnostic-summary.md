# Baseline-only host-submission timing diagnosis

Same A binaries and inputs; this compares timing scopes, not patch speedup. Each scope has one fresh worker per workload and is descriptive only.

| Workload | Consumer/width | Host100 median µs | Graph100 median µs | Host100 sample CV | Graph100 sample CV |
| :--- | :--- | ---: | ---: | ---: | ---: |
| retained | normalization/32 | 7.226 | 4.696 | 0.0403 | 0.0041 |
| retained | first-thread/32 | 7.259 | 4.465 | 0.0361 | 0.0092 |
| retained | normalization/128 | 8.711 | 7.283 | 0.0528 | 0.0165 |
| retained | first-thread/128 | 8.480 | 6.145 | 0.0237 | 0.0044 |
| retained | normalization/256 | 14.506 | 13.303 | 0.0048 | 0.0255 |
| retained | first-thread/256 | 12.438 | 10.696 | 0.0137 | 0.0063 |
| retained | normalization/512 | 28.792 | 27.575 | 0.0031 | 0.0025 |
| retained | first-thread/512 | 22.727 | 21.727 | 0.0055 | 0.0022 |
| integer | normalization/32 | 23.521 | 4.588 | 0.5014 | 0.0993 |
| integer | first-thread/32 | 7.528 | 4.423 | 0.0385 | 0.0143 |
| integer | normalization/128 | 8.206 | 5.776 | 0.1182 | 0.0070 |
| integer | first-thread/128 | 7.269 | 4.486 | 0.0410 | 0.0104 |
| integer | normalization/256 | 12.597 | 11.114 | 0.2745 | 0.0045 |
| integer | first-thread/256 | 8.817 | 7.290 | 0.0577 | 0.0042 |
| integer | normalization/512 | 24.659 | 22.810 | 0.0013 | 0.0070 |
| integer | first-thread/512 | 16.450 | 14.661 | 0.0180 | 0.0045 |
| legacy | normalization/32 | 7.443 | 4.695 | 0.0510 | 0.0083 |
| legacy | first-thread/32 | 7.521 | 4.465 | 0.0865 | 0.0118 |
| legacy | normalization/128 | 8.489 | 7.279 | 0.0359 | 0.0049 |
| legacy | first-thread/128 | 8.595 | 6.146 | 0.3995 | 0.0330 |
| legacy | normalization/256 | 14.502 | 13.294 | 0.0037 | 0.0029 |
| legacy | first-thread/256 | 12.557 | 10.701 | 0.0197 | 0.0018 |
| legacy | normalization/512 | 28.800 | 27.583 | 0.0013 | 0.0040 |
| legacy | first-thread/512 | 22.786 | 21.758 | 0.0073 | 0.0059 |
