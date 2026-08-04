# ZGWT 纯狗资产工具

`generate_lightweight_urdf.py` 从原始 `zgwt.urdf` 生成
`zgwt_lightweight.urdf`。轻量版本完整保留 inertial、joint、collision、关节限位
和其他非 visual 元素，只把约 123 MB 的 STL visual 替换为对应 collision 的
box/cylinder primitive。

四个没有 collision 的 ABAD link 使用仅用于显示的 0.08 m box；这些 box 不会写入
collision，因此不参与接触求解。`zgwt_dance` 使用轻量版本，基础 `zgwt` 任务仍使用
原始完整外观资产，带臂任务继续使用独立的 `zgwsarm.urdf`。

原始 URDF 发生变化后重新生成：

```bash
python tools/zgwt/generate_lightweight_urdf.py
```

物理等价性由 `legged_gym/tests/test_zgwt_lightweight_asset.py` 检查。
