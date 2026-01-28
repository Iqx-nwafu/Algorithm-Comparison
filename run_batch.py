import subprocess
import sys
import os

# ================= 配置区域 =================
# 脚本名称
SCRIPT_NAME = "morl_irrigation_dmoqn.py"

# 模型路径
# (请注意：根据你的报错信息，你的模型可能在 'runs/best_model.pt' 而不是 'runs/dmoqn/best_model.pt')
# 请检查你的文件管理器，确保此路径指向真实的 .pt 文件
MODEL_PATH = "best_model.pt"

# 想要跑的三个 H0 值
H0_VALUES = [23.75, 25.0, 26.25]

# 其他固定参数
NODES = "Nodes.xlsx"
PIPES = "Pipes.xlsx"
ROOT = "J0"
HMIN = 11.59  # <--- [新增] 必须指定 Hmin，这里默认设为 11.59
GRID = "200"


# ===========================================

def run_task(h0):
    print(f"\n{'=' * 40}")
    print(f"正在启动任务: H0 = {h0}m")
    print(f"{'=' * 40}")

    out_dir = f"runs/sweep_{h0}"

    cmd = [
        sys.executable, SCRIPT_NAME, "pareto_sweep",
        "--model", MODEL_PATH,
        "--nodes", NODES,
        "--pipes", PIPES,
        "--root", ROOT,
        "--H0", str(h0),
        "--Hmin", str(HMIN),  # <--- [修正] 添加了 Hmin 参数
        "--scenario_k", "1",
        "--scenario_rel", "1.0",
        "--grid", GRID,
        "--out", out_dir
    ]

    try:
        # 打印一下实际执行的命令，方便调试
        # print("Executing:", " ".join(cmd))

        subprocess.run(cmd, check=True)
        print(f"--> H0={h0} 任务完成！结果已保存在 {out_dir}")
    except subprocess.CalledProcessError as e:
        print(f"!! 任务 H0={h0} 失败，错误码: {e.returncode}")
        # 如果是路径错误或参数错误，通常会在这里停止
        raise


if __name__ == "__main__":
    # 确保 runs 文件夹存在
    os.makedirs("runs", exist_ok=True)

    # 检查模型文件是否存在，避免白跑
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件: {MODEL_PATH}")
        print("请修改代码中的 MODEL_PATH 变量。")
        sys.exit(1)

    for val in H0_VALUES:
        run_task(val)

    print(f"\n{'=' * 40}")
    print("所有批处理任务已全部结束。")