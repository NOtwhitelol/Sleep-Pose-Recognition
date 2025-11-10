import os
import re

folder = "data/generated_images/stomach"        # 資料夾名稱
prefix = "stomach"        # 檔名前綴（會變成 xxx_1.png, xxx_2.png, ...）

pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.png$", re.IGNORECASE)

# 取得所有 PNG 檔案
files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]

# 分出已命名與未命名的檔案
named_files = []
unnamed_files = []
for f in files:
    match = pattern.match(f)
    if match:
        named_files.append((int(match.group(1)), f))
    else:
        unnamed_files.append(f)

# 取得已使用的編號
used_numbers = sorted([num for num, _ in named_files])

# 建立未使用編號清單
available_numbers = []
for n in range(1, len(files) + 1):
    if n not in used_numbers:
        available_numbers.append(n)

# 若有未命名的檔案，依序補上缺少的編號
for f in sorted(unnamed_files):
    if not available_numbers:
        break
    new_num = available_numbers.pop(0)
    old_path = os.path.join(folder, f)
    new_name = f"{prefix}_{new_num}.png"
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"✅ {f} → {new_name}")

# 檢查跳號（例如 prefix_1, prefix_3, prefix_4 → 改成 prefix_1, prefix_2, prefix_3）
named_files = [(int(pattern.match(f).group(1)), f) for f in os.listdir(folder) if pattern.match(f)]
named_files.sort()

expected_num = 1
for actual_num, filename in named_files:
    if actual_num != expected_num:
        old_path = os.path.join(folder, filename)
        new_name = f"{prefix}_{expected_num}.png"
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"🔄 修正跳號：{filename} → {new_name}")
    expected_num += 1

print(f"✅ 所有圖片命名完成，檔名前綴為「{prefix}」！")
