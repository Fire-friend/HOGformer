import os

path = "/data1/wjw/all_in_one_set2/Test/derain/Rain100L/target/"  # ← 修改为你的实际目录
files = os.listdir(path)

for f in files:
    old = os.path.join(path, f)
    new_name = f.replace("norain", "rain")
    new = os.path.join(path, new_name)

    # 只在名称发生变化时重命名
    if old != new:
        # 防止文件重名导致报错
        if not os.path.exists(new):
            os.rename(old, new)
            print(f"Renamed: {f} → {new_name}")
        else:
            print(f"Skipped (target exists): {new_name}")
