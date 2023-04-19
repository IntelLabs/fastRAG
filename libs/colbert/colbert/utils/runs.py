import os
import sys
import time
import traceback
from contextlib import contextmanager

import __main__
import colbert.utils.distributed as distributed
from colbert.utils.logging import Logger
from colbert.utils.utils import create_directory, print_message, timestamp

# import mlflow


class _RunManager:
    def __init__(self):
        self.experiments_root = None
        self.experiment = None
        self.path = None
        self.script = self._get_script_name()
        self.name = self._generate_default_run_name()
        self.original_name = self.name
        self.exit_status = "FINISHED"

        self._logger = None
        self.start_time = time.time()

    def init(self, rank, root, experiment, name):
        assert "/" not in experiment, experiment
        assert "/" not in name, name

        self.experiments_root = os.path.abspath(root)
        self.experiment = experiment
        self.name = name
        self.path = os.path.join(self.experiments_root, self.experiment, self.script, self.name)

        if rank < 1:
            if os.path.exists(self.path):
                print("\n\n")
                print_message("It seems that ", self.path, " already exists.")
                assert not os.path.exists(self.path), self.path
            else:
                create_directory(self.path)

        distributed.barrier(rank)

        self._logger = Logger(rank, self)
        self._log_args = self._logger._log_args
        self.warn = self._logger.warn
        self.info = self._logger.info
        self.info_all = self._logger.info_all
        self.log_metric = self._logger.log_metric
        self.log_new_artifact = self._logger.log_new_artifact

    def _generate_default_run_name(self):
        return timestamp()

    def _get_script_name(self):
        return os.path.basename(__main__.__file__) if "__file__" in dir(__main__) else "none"

    @contextmanager
    def context(self, consider_failed_if_interrupted=True):
        try:
            yield

        except KeyboardInterrupt as ex:
            print("\n\nInterrupted\n\n")
            self._logger._log_exception(ex.__class__, ex, ex.__traceback__)
            self._logger._log_all_artifacts()

            if consider_failed_if_interrupted:
                self.exit_status = "KILLED"  # mlflow.entities.RunStatus.KILLED

            sys.exit(128 + 2)

        except Exception as ex:
            self._logger._log_exception(ex.__class__, ex, ex.__traceback__)
            self._logger._log_all_artifacts()

            self.exit_status = "FAILED"  # mlflow.entities.RunStatus.FAILED

            raise ex

        finally:
            total_seconds = str(time.time() - self.start_time) + "\n"
            original_name = str(self.original_name)
            name = str(self.name)

            self.log_new_artifact(
                os.path.join(self._logger.logs_path, "elapsed.txt"), total_seconds
            )
            self.log_new_artifact(
                os.path.join(self._logger.logs_path, "name.original.txt"), original_name
            )
            self.log_new_artifact(os.path.join(self._logger.logs_path, "name.txt"), name)

            self._logger._log_all_artifacts()

            # mlflow.end_run(status=self.exit_status)


Run = _RunManager()
