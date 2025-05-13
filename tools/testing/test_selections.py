import os
import subprocess

from typing import Callable, Dict, List, Optional, Tuple

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests

IS_MEM_LEAK_CHECK = os.getenv("PYTORCH_TEST_CUDA_MEM_LEAK_CHECK", "0") == "1"

NUM_PROCS = 1 if IS_MEM_LEAK_CHECK else 2

# See Note [ROCm parallel CI testing]
# Special logic for ROCm GHA runners to query number of GPUs available.
# torch.version.hip was not available to check if this was a ROCm self-hosted runner.
# Must check for ROCm runner in another way. We look for /opt/rocm directory.
if os.path.exists("/opt/rocm") and not IS_MEM_LEAK_CHECK and os.environ.get("MACA_PATH") is None:
    try:
        # This is the same logic used in GHA health check, see .github/templates/common.yml.j2
        lines = (
            subprocess.check_output(["rocminfo"], encoding="ascii").strip().split("\n")
        )
        count = 0
        for line in lines:
            if " gfx" in line:
                count += 1
        assert count > 0  # there must be at least 1 GPU
        NUM_PROCS = count
    except subprocess.CalledProcessError as e:
        # The safe default for ROCm GHA runners is to run tests serially.
        NUM_PROCS = 1


class ShardJob:
    def __init__(self, test_times: Dict[str, float]):
        self.test_times = test_times
        self.serial: List[str] = []
        self.parallel: List[str] = []

    def get_total_time(self) -> float:
        procs = [0.0 for _ in range(NUM_PROCS)]
        for test in self.parallel:
            test_time = self.test_times.get(test, 0)
            min_index = procs.index(min(procs))
            procs[min_index] += test_time
        time = max(procs) + sum(self.test_times.get(test, 0) for test in self.serial)
        return time

    def convert_to_tuple(self) -> Tuple[float, List[str]]:
        return (self.get_total_time(), self.serial + self.parallel)


def calculate_shards(
    num_shards: int,
    tests: List[str],
    test_file_times: Dict[str, float],
    must_serial: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[float, List[str]]]:
    must_serial = must_serial or (lambda x: True)

    known_tests = [x for x in tests if x in test_file_times]
    unknown_tests: List[str] = [x for x in tests if x not in known_tests]

    sorted_tests = sorted(known_tests, key=lambda j: test_file_times[j], reverse=True)

    sharded_jobs: List[ShardJob] = [
        ShardJob(test_file_times) for _ in range(num_shards)
    ]
    for test in sorted_tests:
        if must_serial(test):
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.serial.append(test)
        else:
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.parallel.append(test)

    # Round robin the unknown jobs starting with the smallest shard
    index = min(range(num_shards), key=lambda i: sharded_jobs[i].get_total_time())
    for test in unknown_tests:
        sharded_jobs[index].serial.append(test)
        index = (index + 1) % num_shards
    return [job.convert_to_tuple() for job in sharded_jobs]


def _query_changed_test_files() -> List[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'master')}"
    cmd = ["git", "diff", "--name-only", default_branch, "HEAD"]
    proc = subprocess.run(cmd, capture_output=True)

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    return lines


def get_reordered_tests(tests: List[str]) -> List[str]:
    """Get the reordered test filename list based on github PR history or git changed file."""
    prioritized_tests: List[str] = []
    if len(prioritized_tests) == 0:
        try:
            changed_files = _query_changed_test_files()
        except Exception:
            # If unable to get changed files from git, quit without doing any sorting
            return tests

        prefix = f"test{os.path.sep}"
        prioritized_tests = [
            f for f in changed_files if f.startswith(prefix) and f.endswith(".py")
        ]
        prioritized_tests = [f[len(prefix) :] for f in prioritized_tests]
        prioritized_tests = [f[: -len(".py")] for f in prioritized_tests]
        print("Prioritized test from test file changes.")

    bring_to_front = []
    the_rest = []

    for test in tests:
        if test in prioritized_tests:
            bring_to_front.append(test)
        else:
            the_rest.append(test)
    if len(tests) == len(bring_to_front) + len(the_rest):
        print(
            f"reordering tests for PR:\n"
            f"prioritized: {bring_to_front}\nthe rest: {the_rest}\n"
        )
        return bring_to_front + the_rest
    else:
        print(
            f"Something went wrong in CI reordering, expecting total of {len(tests)}:\n"
            f"but found prioritized: {len(bring_to_front)}\nthe rest: {len(the_rest)}\n"
        )
        return tests


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)


###################################### maca ######################################
try:
    import boto3  # type: ignore[import]
    import botocore  # type: ignore[import]
    HAVE_BOTO3 = True
except ImportError:
    HAVE_BOTO3 = False


def _get_stripped_CI_job() -> str:
    """E.g. convert 'pytorch_windows_vs2019_py36_cuda10.1_build' to 'pytorch_windows_vs2019_py36_cuda10.1'.
    """
    job = os.environ.get("JOB_BASE_NAME", "").rstrip('0123456789')
    if job.endswith('_slow_test'):
        job = job[:len(job) - len('_slow_test')]
    elif job.endswith('_test') or job.endswith('-test'):
        job = job[:len(job) - len('_test')]
    elif job.endswith('_build') or job.endswith('-build'):
        job = job[:len(job) - len('_build')]
    return job

import csv
from typing import Dict, List, Optional, Tuple, Union, Any, cast
from typing_extensions import Literal, TypedDict
from datetime import datetime, timedelta
import logging
from collections import defaultdict


Commit = str  # 40-digit SHA-1 hex string
Status = Optional[Literal['errored', 'failed', 'skipped']]
logger = logging.getLogger(__name__)
OSSCI_METRICS_BUCKET = 'ossci-metrics'
import bz2
import json


class JobTimeJSON(TypedDict):
    commit: str
    JOB_BASE_NAME: str
    job_times: Dict[str, float]

class ReportMetaMeta(TypedDict):
    build_pr: str
    build_tag: str
    build_sha1: Commit
    build_base_commit: Commit
    build_branch: str
    build_job: str
    build_workflow_id: str
    build_start_time_epoch: str

class CaseMeta(TypedDict):
    seconds: float

class Version1Case(CaseMeta):
    name: str
    errored: bool
    failed: bool
    skipped: bool

class Version2Case(CaseMeta):
    status: Status

class Version2Suite(TypedDict):
    total_seconds: float
    cases: Dict[str, Version2Case]

class Version2File(TypedDict):
    total_seconds: float
    suites: Dict[str, Version2Suite]

class ReportMeta(ReportMetaMeta):
    total_seconds: float

class VersionedReport(ReportMeta):
    format_version: int

class Version1Suite(TypedDict):
    total_seconds: float
    cases: List[Version1Case]

class Version1Report(ReportMeta):
    suites: Dict[str, Version1Suite]

class Version2Report(VersionedReport):
    files: Dict[str, Version2File]

class VersionedReport(ReportMeta):
    format_version: int

def _get_job_times_json(job_times: Dict[str, float]) -> JobTimeJSON:
    return {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], encoding="ascii").strip(),
        'JOB_BASE_NAME': _get_stripped_CI_job(),
        'job_times': job_times,
    }


def _calculate_job_times(reports: List["Report"]) -> Dict[str, float]:
    """Compute test runtime by filename: ("test_file_name" -> (current_avg, # values))
    """
    jobs_to_times: Dict[str, Tuple[float, int]] = dict()
    for report in reports:
        v_report = cast(Version2Report, report)
        assert 'format_version' in v_report.keys() and v_report.get('format_version') == 2, \
            "S3 format currently handled is version 2 only"
        files: Dict[str, Any] = v_report['files']
        for name, test_file in files.items():
            if name not in jobs_to_times:
                jobs_to_times[name] = (test_file['total_seconds'], 1)
            else:
                curr_avg, curr_count = jobs_to_times[name]
                new_count = curr_count + 1
                new_avg = (curr_avg * curr_count + test_file['total_seconds']) / new_count
                jobs_to_times[name] = (new_avg, new_count)

    return {job: time for job, (time, _) in jobs_to_times.items()}

Report = Union[Version1Report, VersionedReport]
if HAVE_BOTO3:
    S3_RESOURCE_READ_ONLY = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    S3_RESOURCE = boto3.resource('s3')

def get_S3_bucket_readonly(bucket_name: str) -> Any:
    return S3_RESOURCE_READ_ONLY.Bucket(bucket_name)

def _parse_master_summaries(summaries: Any, jobs: List[str]) -> Dict[str, List[Report]]:
    summary_dict = defaultdict(list)
    for summary in summaries:
        # master summary format: "test_time/{sha}/{job}/file"
        summary_job = summary.key.split('/')[2]
        if summary_job in jobs or len(jobs) == 0:
            binary = summary.get()["Body"].read()
            string = bz2.decompress(binary).decode("utf-8")
            summary_dict[summary_job].append(json.loads(string))
    return summary_dict

def get_test_stats_summaries(*, sha: str, jobs: Optional[List[str]] = None) -> Dict[str, List[Report]]:
    bucket = get_S3_bucket_readonly(OSSCI_METRICS_BUCKET)
    summaries = bucket.objects.filter(Prefix=f"test_time/{sha}")
    return _parse_master_summaries(summaries, jobs=list(jobs or []))

def get_test_stats_summaries_for_job(*, sha: str, job_prefix: str) -> Dict[str, List[Report]]:
    bucket = get_S3_bucket_readonly(OSSCI_METRICS_BUCKET)
    summaries = bucket.objects.filter(Prefix=f"test_time/{sha}/{job_prefix}")
    return _parse_master_summaries(summaries, jobs=list())

def get_previous_reports_for_branch(branch: str, ci_job_prefix: str = "") -> List[Report]:
    commit_date_ts = subprocess.check_output(
        ['git', 'show', '-s', '--format=%ct', 'HEAD'],
        encoding="ascii").strip()
    commit_date = datetime.fromtimestamp(int(commit_date_ts))
    # We go a day before this current commit to avoiding pulling incomplete reports
    day_before_commit = str(commit_date - timedelta(days=1)).split(' ')[0]
    # something like git rev-list --before="2021-03-04" --max-count=10 --remotes="*origin/nightly"
    commits = subprocess.check_output(
        ["git", "rev-list", f"--before={day_before_commit}", "--max-count=10", f"--remotes=*{branch}"],
        encoding="ascii").splitlines()

    reports: List[Report] = []
    commit_index = 0
    while len(reports) == 0 and commit_index < len(commits):
        commit = commits[commit_index]
        logger.info(f'Grabbing reports from commit: {commit}')
        summaries = get_test_stats_summaries_for_job(sha=commit, job_prefix=ci_job_prefix)
        for job_name, summary in summaries.items():
            reports.append(summary[0])
            if len(summary) > 1:
                logger.warning(f'WARNING: Multiple summary objects found for {commit}/{job_name}')
        commit_index += 1
    return reports


def _pull_job_times_from_S3() -> Dict[str, float]:
    if HAVE_BOTO3:
        ci_job_prefix = _get_stripped_CI_job()
        s3_reports: List["Report"] = get_previous_reports_for_branch('origin/viable/strict', ci_job_prefix)
    else:
        print('Uh oh, boto3 is not found. Either it is not installed or we failed to import s3_stat_parser.')
        print('If not installed, please install boto3 for automatic sharding and test categorization.')
        s3_reports = []

    if len(s3_reports) == 0:
        print('Gathered no reports from S3. Please proceed without them.')
        return dict()

    return _calculate_job_times(s3_reports)


def export_S3_test_times(test_times_filename: Optional[str] = None) -> Dict[str, float]:
    test_times: Dict[str, float] = _pull_job_times_from_S3()
    if test_times_filename is not None:
        print(f'Exporting S3 test stats to {test_times_filename}.')
        if os.path.exists(test_times_filename):
            print(f'Overwriting existent file: {test_times_filename}')
        with open(test_times_filename, 'w+') as file:
            job_times_json = _get_job_times_json(test_times)
            json.dump(job_times_json, file, indent='    ', separators=(',', ': '))
            file.write('\n')
    return test_times


def get_specified_test_cases(filename: str, tests: List[str]) -> Dict[str, List[str]]:
    """Get test cases from a specified test case file. Usually exported manually or through CI system.
    """
    if not os.path.exists(filename):
        print(f'Could not find specified tests file: {filename}. Proceeding with default behavior.')
        return dict()

    # The below encoding is utf-8-sig because utf-8 doesn't properly handle the byte-order-mark character
    with open(filename, mode='r', encoding="utf-8-sig") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        specified_test_case_dict: Dict[str, List[str]] = dict()
        for row in csv_reader:
            line_count += 1
            if line_count == 1:
                if 'test_filename' not in row or 'test_case_name' not in row:
                    print('Data is missing necessary columns for test specification. Proceeding with default behavior.')
                    return dict()
            test_filename = row['test_filename']
            if test_filename.startswith('#'):
                continue
            test_case_name = row['test_case_name']
            if test_filename not in tests:
                print(f'Specified test_filename {test_filename} not found in TESTS. Skipping.')
                continue
            if test_filename not in specified_test_case_dict:
                specified_test_case_dict[test_filename] = []
            specified_test_case_dict[test_filename].append(test_case_name)
        print(f'Processed {line_count} test cases.')
        return specified_test_case_dict
###################################### maca ######################################
