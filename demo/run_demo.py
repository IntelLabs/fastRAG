import argparse
import os
import subprocess  # nosec B404
import time

TASKS = {
    "QA1": "qa_pipeline.yaml",
    "QA2": "qa_plaid.yaml",
    "QADIFF": "qa_diffusion_pipeline.yaml",
    "SUMR": "summarization_pipeline.yaml",
    "LLM": "rag_generation_with_dynamic_prompt.yaml",
}

SCREENS = {
    "QA1": "webapp",
    "QA2": "webapp",
    "QADIFF": "webapp",
    "SUMR": "webapp_summarization",
    "LLM": "prompt_llm",
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
    parser.add_argument(
        "-e", "--endpoint", default="http://localhost:8000", help="pipeline service endpoint"
    )
    parser.add_argument(
        "--only-ui",
        action="store_true",
        help="launch only the UI interface (without launching a service)",
    )

    args = parser.parse_args()
    path = os.getcwd()

    s_pid = "NA"
    if not args.only_ui:
        # Create REST server
        cmd = f"python -m fastrag.rest_api.application --config={path}/config/TASKCONFIGURATION"
        cmd = cmd.replace("TASKCONFIGURATION", TASKS[args.task_config])
        print("Launching fastRAG pipeline service...")
        run_service(cmd)
        time.sleep(10)
        s_pid = get_pid("fastrag.rest_api.application")

    # Create UI
    os.environ["API_ENDPOINT"] = f"{args.endpoint}"
    cmd = f"python -m streamlit run {path}/fastrag/ui/SCREEN.py"
    cmd = cmd.replace("SCREEN", SCREENS[args.task_config])
    print("Launching UI...")
    time.sleep(3)
    run_service(cmd)
    u_pid = get_pid("streamlit run")

    print("\n")
    print(f"Server on  {args.endpoint}/docs  PID={s_pid}")
    print(f"UI on      localhost:8501        PID={u_pid}")
