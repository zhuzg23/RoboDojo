#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import os
import shutil
import socket
import subprocess
import sys
import time


# ============================================================
# 基本配置
# ============================================================

ROBODOJO_DIR = Path.home() / "projects" / "RoboDojo"

POLICY_DIR = (
    Path.home()
    / "projects"
    / "XPolicyLab-pi05"
    / "policy"
    / "Pi_05"
)

TASK = "stack_bowls"
ENV_CFG = "arx_x5"

CKPT = "RoboDojo-sim-arx_x5-joint-0"

ACTION_TYPE = "joint"

POLICY_HOST = "127.0.0.1"
POLICY_PORT = 6000

SEED = 0
ENV_GPU = 0
EVAL_NUM = 1


# ============================================================
# 工具函数
# ============================================================

def check_policy_server():
    """
    检查 Pi0.5 websocket policy server 是否正在监听 6000 端口。
    """

    print()
    print("===== CHECK PI0.5 SERVER =====")

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    sock.settimeout(2.0)

    try:
        result = sock.connect_ex(
            (
                POLICY_HOST,
                POLICY_PORT,
            )
        )
    finally:
        sock.close()

    if result != 0:
        print(
            f"[FAIL] Pi0.5 server 没有监听 "
            f"{POLICY_HOST}:{POLICY_PORT}"
        )
        print()
        print(
            "请先把 Pi0.5 policy server 启动起来，"
            "并保持那个终端一直开着。"
        )
        sys.exit(1)

    print(
        f"[PASS] Pi0.5 server 正常："
        f"{POLICY_HOST}:{POLICY_PORT}"
    )


def find_latest_eval_dir(start_time):
    """
    找到本次评测刚刚生成的 RoboDojo 结果目录。
    """

    base = (
        ROBODOJO_DIR
        / "eval_result"
        / "RoboDojo"
        / TASK
        / "Pi_05"
        / ENV_CFG
    )

    if not base.exists():
        return None

    candidates = []

    for path in base.rglob("*"):
        if not path.is_dir():
            continue

        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue

        # 只考虑本次程序启动以后附近生成/修改的目录
        if mtime >= start_time - 5:
            candidates.append(
                (
                    mtime,
                    path,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    return candidates[-1][1]


def copy_media(eval_dir, output_dir):
    """
    把本次评测生成的 mp4/png 复制到统一输出文件夹。
    """

    if eval_dir is None:
        print(
            "[WARN] 没找到新的 eval_result 目录。"
        )
        return []

    media_files = []

    for pattern in (
        "*.mp4",
        "*.png",
    ):
        for src in eval_dir.glob(pattern):

            dst = output_dir / src.name

            shutil.copy2(
                src,
                dst
            )

            media_files.append(dst)

    return media_files


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("========================================")
    print("  RoboDojo + Pi0.5 EGL 三相机评测")
    print("========================================")

    # --------------------------------------------------------
    # 1. 检查路径
    # --------------------------------------------------------

    if not ROBODOJO_DIR.exists():
        print(
            f"[FAIL] RoboDojo 不存在："
            f"{ROBODOJO_DIR}"
        )
        sys.exit(1)

    if not POLICY_DIR.exists():
        print(
            f"[FAIL] Pi0.5 policy 目录不存在："
            f"{POLICY_DIR}"
        )
        sys.exit(1)

    check_policy_server()

    # --------------------------------------------------------
    # 2. 创建本次独立输出目录
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        ROBODOJO_DIR
        / "test_outputs"
        / f"pi05_threecam_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_path = (
        output_dir
        / "eval.log"
    )

    print()
    print("===== OUTPUT =====")
    print(
        f"本次结果目录：\n{output_dir}"
    )

    # --------------------------------------------------------
    # 3. 环境变量
    # --------------------------------------------------------

    env = os.environ.copy()

    # 我们已经验证成功的 RTX 绕过方案
    env[
        "ROBODOJO_NO_RTX_CAMERA"
    ] = "1"

    env[
        "ROBODOJO_EGL_LIVE"
    ] = "1"

    # 关闭之前的临时测试模式
    env[
        "ROBODOJO_EGL_QUICK_WRIST"
    ] = "0"

    env[
        "ROBODOJO_EGL_OBJECT_TEST"
    ] = "0"

    # Headless
    env["HEADLESS"] = "1"
    env["ENABLE_CAMERAS"] = "0"
    env["LIVESTREAM"] = "0"

    env["EVAL_NUM"] = str(
        EVAL_NUM
    )

    # --------------------------------------------------------
    # 4. 构造正式评测命令
    # --------------------------------------------------------

    command = [
        "bash",
        "scripts/robodojo.sh",
        "client",

        "--policy-dir",
        str(POLICY_DIR),

        "--task",
        TASK,

        "--env-cfg",
        ENV_CFG,

        "--seed",
        str(SEED),

        "--env-gpu",
        str(ENV_GPU),

        "--ckpt",
        CKPT,

        "--action-type",
        ACTION_TYPE,

        "--policy-host",
        POLICY_HOST,

        "--policy-port",
        str(POLICY_PORT),

        "--eval-num",
        str(EVAL_NUM),
    ]

    print()
    print("===== START EVALUATION =====")
    print(
        f"Task       : {TASK}"
    )
    print(
        f"Environment: {ENV_CFG}"
    )
    print(
        f"Checkpoint : {CKPT}"
    )
    print(
        "Camera     : "
        "cam_head + cam_left_wrist + cam_right_wrist"
    )
    print(
        "Renderer   : EGL (RTX bypass)"
    )
    print()

    start_time = time.time()

    # --------------------------------------------------------
    # 5. source RoboDojo 环境，然后运行
    # --------------------------------------------------------

    shell_command = (
        "source /opt/robodojo/activate.sh"
        " && "
        + " ".join(
            subprocess.list2cmdline([x])
            for x in command
        )
    )

    with log_path.open(
        "w",
        encoding="utf-8"
    ) as log_file:

        process = subprocess.Popen(
            [
                "bash",
                "-lc",
                shell_command,
            ],
            cwd=ROBODOJO_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:

            # 终端实时显示
            print(
                line,
                end=""
            )

            # 同时写进 eval.log
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    print()
    print(
        "===== EVALUATION PROCESS FINISHED ====="
    )
    print(
        f"return code = {return_code}"
    )

    # --------------------------------------------------------
    # 6. 找到 RoboDojo 原始结果
    # --------------------------------------------------------

    eval_dir = find_latest_eval_dir(
        start_time
    )

    print()
    print("===== ROBO DOJO RESULT DIR =====")

    if eval_dir is None:

        print(
            "[WARN] 没找到本次 eval_result 目录"
        )

    else:

        print(eval_dir)

    # --------------------------------------------------------
    # 7. 把三个视频复制进统一输出目录
    # --------------------------------------------------------

    media_files = copy_media(
        eval_dir,
        output_dir
    )

    print()
    print("========================================")
    print("              RESULT")
    print("========================================")

    print()
    print(
        f"完整日志：\n{log_path}"
    )

    if media_files:

        print()
        print("本次图片/视频：")

        for path in media_files:
            print(
                f"  {path.name}"
            )

    else:

        print()
        print(
            "[WARN] 没有找到 mp4/png 文件。"
        )

    # --------------------------------------------------------
    # 8. 从日志中提取最终成功/失败统计
    # --------------------------------------------------------

    try:
        text = log_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        summary_lines = [
            line
            for line in text.splitlines()
            if (
                "Success nums:" in line
                or "Fail nums:" in line
                or "Unstable nums:" in line
            )
        ]

        if summary_lines:

            print()
            print(
                "===== EVALUATION SUMMARY ====="
            )

            for line in summary_lines[-5:]:
                print(line)

    except Exception as exc:

        print(
            f"[WARN] 读取 summary 失败：{exc}"
        )

    print()
    print("========================================")
    print("本次所有整理后的结果都在：")
    print(output_dir)
    print("========================================")
    print()


if __name__ == "__main__":
    main()
