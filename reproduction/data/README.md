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

## 已完成的输入预检

三个 target config 已实际读取一个 `batch=64` 的 RLDS batch，并通过完整 transform chain：state `[64, 32]`、action `[64, 50, 32]`、三路图像 `[64, 224, 224, 3]`、prompt tokens `[64, 48]`，数值均为有限 float32。机器可读记录见 `../experiments/input-smoke-targets.json`。

预检时发现公开 registry 遗漏 mugs 与 basket 的 dataset name；本复现只补齐与 stove 相同的 schema mapping，不改变样本、采样权重或 transform 算法。

## 传输恢复证据

- `retain_dataset_manifest_sha.json`：固定 revision 的 353 文件 size/SHA-256 参考。
- `transfer-audit-retry.json`：首轮传输停止后的服务器尺寸审计，确认 271 个文件完整、82 个 TFRecord 待续传。
- `retry-selection-20260820.json`：82 个续传对象及总字节数，不包含会过期的签名 URL。
