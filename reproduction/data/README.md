# 数据清单

官方数据仓库：`yajatyadav/RETAIN_datasets`

固定 revision：`d15edfa89167e6e7230be3e85eb7391be2fa3134`

服务器目标目录：`/shared/.cache/retain/libero/datasets`

公开文件元数据（过滤后）：

| Dataset | Trajectories | Transitions |
|---|---:|---:|
| `libero_goal_reduced` | 349 | 39,799 |
| `libero_object_reduced` | 365 | 53,739 |
| `libero_spatial_reduced` | 343 | 42,391 |
| `libero_90_flipped` | 3,917 | 567,494 |
| stove target | 41 | 10,866 |
| mugs target | 38 | 9,807 |
| basket target | 43 | 11,494 |

下载完成后，本目录将补充：

- 文件级 SHA-256 清单；
- 总字节数和文件数；
- 7 个 RLDS dataset directory 的 `dataset_info.json` 摘要；
- dataloader 抽样后的 tensor shape / dtype / prompt 检查。
