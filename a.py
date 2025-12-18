import numpy as np
import os
import shutil

# ==============================================================================
# 1. 配置参数 (请在这里修改)
# ==============================================================================

# 你想要修改的 .npy 文件的完整路径
# 注意：Windows路径前的 r'' 很重要，可以防止转义字符问题
FILE_PATH = r'D:\tie-smac_IMPLUSE\date\w_env_8m_seed_2025_win_rates.npy'

# --- 选择你的操作模式 ---
MODE = 'change_win_rate'  # 模式一：修改整体胜率
# MODE = 'set_first_element_to_one'  # 模式二：将第一个元素设置为1

# --- 模式一的参数 (仅当 MODE = 'change_win_rate' 时有效) ---
TARGET_WIN_RATE = 1


# ==============================================================================
# 2. 功能函数
# ==============================================================================

def change_win_rate_logic(data, target_rate):
    """
    加载一个npy数组，并将其胜率修改为目标值。
    """
    total_elements = len(data)
    target_wins = round(total_elements * target_rate)
    current_wins = np.sum(data)

    print(f"\n--- 模式: 修改胜率 ---")
    print(f"总长度: {total_elements}")
    print(f"当前胜/负: {current_wins} / {total_elements - current_wins}")
    print(f"当前胜率: {current_wins / total_elements:.2%}")
    print(f"目标胜率: {target_rate:.2%}")
    print(f"目标胜利局数: {target_wins}")

    num_to_change = int(abs(target_wins - current_wins))

    if num_to_change == 0:
        print("\n--- 无需修改 ---\n当前胜率已达到目标值。")
        return data, False  # 返回原始数据和“未修改”标记

    if current_wins < target_wins:
        print(f"\n需要增加 {num_to_change} 场胜利。")
        zero_indices = np.where(data == 0)[0]
        np.random.shuffle(zero_indices)
        indices_to_flip = zero_indices[:num_to_change]
        data[indices_to_flip] = 1
        print(f"已将 {len(indices_to_flip)} 个 0 随机修改为 1。")

    elif current_wins > target_wins:
        print(f"\n需要减少 {num_to_change} 场胜利。")
        one_indices = np.where(data == 1)[0]
        np.random.shuffle(one_indices)
        indices_to_flip = one_indices[:num_to_change]
        data[indices_to_flip] = 0
        print(f"已将 {len(indices_to_flip)} 个 1 随机修改为 0。")

    return data, True  # 返回修改后的数据和“已修改”标记


def set_first_element_logic(data):
    """
    加载一个npy数组，并将其第一个元素强制修改为1。
    """
    print(f"\n--- 模式: 设置第一个元素为 1 ---")

    if len(data) == 0:
        print("错误: 数组为空，无法修改。")
        return data, False

    original_value = data[0]
    print(f"第一个元素的原始值: {original_value}")

    if original_value == 1:
        print("\n--- 无需修改 ---\n第一个元素的值已经是 1。")
        return data, False

    data[0] = 1
    print(f"已将第一个元素的值从 {original_value} 修改为 1。")
    return data, True


# ==============================================================================
# 3. 主执行逻辑
# ==============================================================================

def main_modifier(file_path, mode, **kwargs):
    """
    主函数，根据选择的模式加载、修改并保存npy文件。
    """
    # --- 检查文件是否存在 ---
    if not os.path.exists(file_path):
        print(f"!!! 错误: 文件未找到! \n路径: {file_path}")
        return

    # --- 创建备份 ---
    backup_path = file_path + '.bak'
    try:
        shutil.copy2(file_path, backup_path)
        print(f"--- 文件操作 ---")
        print(f"已创建备份文件: {backup_path}")
    except Exception as e:
        print(f"创建备份文件失败: {e}")
        return

    # --- 加载数据 ---
    try:
        original_data = np.load(file_path)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return

    # --- 根据模式执行修改 ---
    if mode == 'change_win_rate':
        target_rate = kwargs.get('target_rate', 0.5)  # 从参数获取目标胜率，默认0.5
        modified_data, was_modified = change_win_rate_logic(original_data.copy(), target_rate)
    elif mode == 'set_first_element_to_one':
        modified_data, was_modified = set_first_element_logic(original_data.copy())
    else:
        print(f"!!! 错误: 未知的模式 '{mode}'。请检查 MODE 配置。")
        return

    # --- 验证并保存 ---
    if was_modified:
        print(f"\n--- 验证结果 ---")
        if mode == 'change_win_rate':
            final_wins = np.sum(modified_data)
            total_elements = len(modified_data)
            print(f"修改后胜/负: {final_wins} / {total_elements - final_wins}")
            print(f"修改后胜率: {final_wins / total_elements:.2%}")
        elif mode == 'set_first_element_to_one':
            print(f"修改后第一个元素的值: {modified_data[0]}")

        np.save(file_path, modified_data)
        print(f"\n[√] 修改成功！新数据已覆盖保存至: {file_path}")
    else:
        # 如果没有修改，删除备份文件
        os.remove(backup_path)
        print(f"已删除未使用的备份文件。")


if __name__ == '__main__':
    # 将配置参数传递给主函数
    main_modifier(FILE_PATH, MODE, target_rate=TARGET_WIN_RATE)