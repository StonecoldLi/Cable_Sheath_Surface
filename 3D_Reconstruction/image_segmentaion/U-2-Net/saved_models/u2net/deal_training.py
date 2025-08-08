import pandas as pd

# 1. 读取 Excel 文件
df = pd.read_excel('./training_losses.xlsx')

# 2. 过滤出 'epoch' 列为 5 的倍数的行
df_filtered = df[(df['iteration'] == 1) | (df['iteration'] % 5 == 0)]

# 3. 保存过滤后的数据到新的 Excel 文件
df_filtered.to_excel('./training_losses_filtered.xlsx', index=False)

print("过滤完成，结果已保存到 'path_to_filtered_file.xlsx'")
