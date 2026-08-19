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

服务器输入门禁已经完成：固定 revision 的 353 个 payload、`24,235,684,869` bytes 与本地参考 SHA-256 全部一致。服务器目录另含 3 个 dataloader smoke test 生成的 `dataset_statistics_*.json` 缓存；它们被保留供训练使用，但不计入固定 revision 的 payload 集合。

## 已完成的输入预检

全部 7 个训练 config 已实际读取一个 `batch=64` 的 RLDS batch，并通过完整 transform chain：state `[64, 32]`、action `[64, 50, 32]`、三路图像 `[64, 224, 224, 3]`、prompt tokens `[64, 48]`，数值均为有限 float32。机器可读记录见 `../experiments/input-smoke-all.json`；前三个 target-only 预检仍保存在 `../experiments/input-smoke-targets.json`。

预检时发现公开 registry 遗漏 mugs 与 basket 的 dataset name；本复现只补齐与 stove 相同的 schema mapping，不改变样本、采样权重或 transform 算法。

## 传输恢复证据

- `retain_dataset_manifest_sha.json`：固定 revision 的 353 文件 size/SHA-256 参考。
- `transfer-audit-retry.json`：首轮传输停止后的服务器尺寸审计，确认 271 个文件完整、82 个 TFRecord 待续传。
- `retry-selection-20260820.json`：82 个续传对象及总字节数，不包含会过期的签名 URL。
- `dataset_manifest_server_failed_20260820T0032.json`：首次全量 SHA 门禁检出的 12 个同尺寸损坏 shard 及其现场哈希。
- `dataset_manifest_server.json`：修复后的服务器实际清单；含 353 个参考 payload 和 3 个派生统计缓存。
- `pi0_base_restore.json`：官方基础权重的 Orbax restore 结构、参数量与结构哈希。
