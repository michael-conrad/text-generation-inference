import os
import time
import traceback

from benchmarks.engine import TGIDockerRunner
from benchmarks.k6 import K6Config, K6Benchmark, K6ConstantArrivalRateExecutor, K6ConstantVUsExecutor, ExecutorInputType
from loguru import logger
import pandas as pd

from parse_load_test import TestType, parse_json_files, plot_metrics


def run_full_test(engine_name: str):
    vus_concurrences = list(range(0, 1024, 40))
    vus_concurrences[0] = 1
    vus_concurrences.append(1024)
    arrival_rates = list(range(0, 200, 10))
    arrival_rates[0] = 1
    arrival_rates.append(200)
    for input_type in [ExecutorInputType.SHAREGPT_CONVERSATIONS, ExecutorInputType.CONSTANT_TOKENS]:
        for c in arrival_rates:
            logger.info(f'Running k6 with constant arrival rate for {c} req/s with input type {input_type.value}')
            k6_executor = K6ConstantArrivalRateExecutor(2000, c, '60s', input_type)
            k6_config = K6Config(f'{engine_name}', k6_executor)
            benchmark = K6Benchmark(k6_config, 'results/')
            benchmark.run()
            return
        for c in vus_concurrences:
            logger.info(f'Running k6 with constant VUs with concurrency {c} with input type {input_type.value}')
            k6_executor = K6ConstantVUsExecutor(c, '60s', input_type)
            k6_config = K6Config(f'{engine_name}', k6_executor)
            benchmark = K6Benchmark(k6_config, 'results/')
            benchmark.run()


def merge_previous_results(csv_path: str, df: pd.DataFrame, version_id: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        previous_df = pd.read_csv(csv_path)
        previous_df['name'] = previous_df['name'].str.replace('tgi', f'tgi_{version_id}')
        df = pd.concat([previous_df, df])
    return df


def main():
    model = 'Qwen/Qwen2-7B'
    runner = TGIDockerRunner(model)
    max_concurrent_requests = 8000
    # run TGI
    try:
        logger.info('Running TGI')
        runner.run([('max-concurrent-requests', max_concurrent_requests)])
        logger.info('TGI is running')
        run_full_test('tgi')
    except Exception as e:
        logger.error(f'Error: {e}')
        # print the stack trace
        print(traceback.format_exc())
    finally:
        runner.stop()
        time.sleep(5)

    for test_type in [TestType.CONSTANT_VUS, TestType.CONSTANT_ARRIVAL_RATE]:
        directory = f'results/{test_type.value.lower()}'
        # check if directory exists
        if not os.path.exists(directory):
            logger.error(f'Directory {directory} does not exist')
            continue
        dfs = parse_json_files(directory, test_type)
        # check if we have previous results CSV file by listing /tmp/artifacts directory, merge them if they exist
        if os.path.exists('/tmp/artifacts'):
            for f in os.listdir('/tmp/artifacts'):
                if f.endswith('.csv'):
                    csv_path = os.path.join('/tmp/artifacts', f)
                    dfs = merge_previous_results(csv_path, dfs, f.split('-')[-1])
        plot_metrics(dfs, test_type, test_type.value.lower())
        # save the data to a csv file
        path = f"{os.getcwd()}/{test_type.value.lower()}.csv"
        dfs.to_csv(f"{path}")


if __name__ == '__main__':
    main()
