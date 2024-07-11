import time
import traceback

from benchmarks.engine import TGIDockerRunner
from benchmarks.k6 import K6Config, K6Benchmark, K6ConstantArrivalRateExecutor, K6ConstantVUsExecutor, ExecutorInputType
from loguru import logger

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
            logger.info(f"Running k6 with constant arrival rate for {c} req/s with input type {input_type.value}")
            k6_executor = K6ConstantArrivalRateExecutor(2000, c, "60s", input_type)
            k6_config = K6Config(f"{engine_name}", k6_executor)
            benchmark = K6Benchmark(k6_config, "results/load_test")
            benchmark.run()
            return
        for c in vus_concurrences:
            logger.info(f"Running k6 with constant VUs with concurrency {c} with input type {input_type.value}")
            k6_executor = K6ConstantVUsExecutor(c, "60s", input_type)
            k6_config = K6Config(f"{engine_name}", k6_executor)
            benchmark = K6Benchmark(k6_config, "results/")
            benchmark.run()


def main():
    model = "Qwen/Qwen2-7B"
    runner = TGIDockerRunner(model)
    max_concurrent_requests = 8000
    # run TGI
    try:
        logger.info("Running TGI")
        runner.run([("max-concurrent-requests", max_concurrent_requests)])
        logger.info("TGI is running")
        run_full_test("tgi")
    except Exception as e:
        logger.error(f"Error: {e}")
        # print the stack trace
        print(traceback.format_exc())
    finally:
        runner.stop()
        time.sleep(5)

    for test_type in [TestType.CONSTANT_VUS, TestType.CONSTANT_ARRIVAL_RATE]:
        directory = f"results/{test_type.value.lower()}"
        dfs = parse_json_files(directory, test_type)
        plot_metrics(dfs, test_type, test_type.value.lower())


if __name__ == '__main__':
    main()
