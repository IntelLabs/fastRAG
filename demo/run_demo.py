import argparse
import os
import subprocess  # nosec B404
import time

TASKS = {
    "QA1": "qa_pipeline.yaml",
    "QA2": "qa_plaid.yaml",
    "QADIFF": "qa_diffusion_pipeline.yaml",
    "SUMR": "summarization_pipeline.yaml",
}

SCREENS = {
    "QA1": "webapp",
    "QA2": "webapp",
    "QADIFF": "webapp",
    "SUMR": "webapp_summarization",
}


def run_service(cmd):
    """Starting a subprocess in the background"""
    subprocess.Popen(f"{cmd} >/dev/null 2>&1 &", shell=True)


def get_pid(cmd):
    output = subprocess.check_output(
        f"ps aux | grep '{cmd}' | grep -v 'grep' | awk '{{print $2}}'", shell=True
    )
    return output.decode("utf-8").strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fastRAG Demo")
    parser.add_argument(
        "-t",
        "--task_config",
        type=str,
        default="QA1",
        choices=list(TASKS.keys()),
        help=f"The abbreviated name for the task configuraion. \n {TASKS} \n",
    )

    args = parser.parse_args()
    path = os.getcwd()

    # Create REST server
    cmd = f"python -m fastrag.rest_api.application --config={path}/config/TASKCONFIGURATION"
    cmd = cmd.replace("TASKCONFIGURATION", TASKS[args.task_config])
    run_service(cmd)

    # Create UI
    os.environ["API_ENDPOINT"] = "http://localhost:8000"
    cmd = f"python -m streamlit run {path}/fastrag/ui/SCREEN.py"
    cmd = cmd.replace("SCREEN", SCREENS[args.task_config])
    run_service(cmd)

    # Sleep and wait for initialization, pids
    print("Creating services...")
    time.sleep(10)
    s_pid = get_pid("fastrag.rest_api.application")
    u_pid = get_pid("streamlit run")

    print("\n")
    print(f"Server on  localhost:8000/docs   PID={s_pid}")
    print(f"UI on      localhost:8501        PID={u_pid}")
