* TOC
{:toc}

## MLRun Setup
Docs: [Set up your client environment](https://docs.mlrun.org/en/latest/install/remote.html)

Create a `mlrun.env` file for environment variables
```
# MLRun DB
MLRUN_DBPATH=<URL endpoint of the MLRun APIs service endpoint; e.g., "https://mlrun-api.default-tenant.app.mycluster.iguazio.com">

# Iguazio filesystem
V3IO_USERNAME=<username of a platform user with access to the MLRun service>
V3IO_ACCESS_KEY=<platform access key>
```

Connect via MLRun Python SDK
```python
# Import MLRun
import mlrun

# Set environment variables - secrets, remote API endpoint, etc.
mlrun.set_env_from_file("mlrun.env")
```

## MLRun Projects
Docs: [Projects and automation](https://docs.mlrun.org/en/latest/projects/project.html)

### General Workflow
Docs: [Create, save, and use projects](https://docs.mlrun.org/en/latest/projects/create-project.html)
```python
# Create or load project
project = mlrun.get_or_create_project(name="my-project", context="./")

# Add function to project
project.set_function(name='train_model', func='train_model.py', kind='job', image='mlrun/mlrun')

# Add workflow (pipeline) to project
project.set_workflow(name='training_pipeline', workflow_path='straining_pipeline.py')

# Save project and generate project.yaml file
project.save()

# Run pipeline via project
project.run(name="training_pipeline", arguments={...})
```

### Git Integration
Docs: [Create and use functions](https://docs.mlrun.org/en/latest/runtimes/create-and-use-functions.html#multiple-source-files)

An MLRun project can be backed by a Git repo. Functions will consume the repo and pull the code either once when Docker image is built (production workflow) or at runtime (development workflow).

Pull repo code once (bake into Docker image)
```python
project.set_source(source="git://github.com/mlrun/project-archive.git")

fn = project.set_function(
    name="myjob", handler="job_func.job_handler",
    image="mlrun/mlrun", kind="job", with_repo=True,
)

project.build_function(fn)
```

Pull repo code at runtime
```python
project.set_source(source="git://github.com/mlrun/project-archive.git", pull_at_runtime=True)

fn = project.set_function(
    name="nuclio", handler="nuclio_func:nuclio_handler",
    image="mlrun/mlrun", kind="nuclio", with_repo=True,
)
```

### CI/CD Integration

#### Overview
Docs: [CI/CD integration](https://docs.mlrun.org/en/latest/projects/ci-integration.html)

Best practice for working with CI/CD is using [MLRun Projects](https://docs.mlrun.org/en/latest/projects/project.html) with a combination of the following:
- **Git:** Single source of truth for source code and deployments via infrastructure as code. Allows for collaboration between multiple developers. An MLRun project can (and should be) be tied to a Git repo. One project maps to one Git repo.
- **CI/CD:** Main tool for orchestrating production deployments. The CI/CD system should be responsible for deploying latest code changes from Git onto the remote cluster via MLRun Python SDK or CLI. 
- **Iguazio/MLRun:** Kubernetes based compute environment for running data analytics, model training, or model deployment tasks.  Additionally, the cluster is where all experiment tracking, job information, logs, and more is located.

See [MLRun Projects](https://docs.mlrun.org/en/latest/projects/project.html) for more information on Git and CI/CD integration. In practice, this may look something like the following:
![](./img/cicd_flow.png)

#### Example (GitHub Actions)
Full Example: [MLRun project-demo](https://github.com/mlrun/project-demo)

```yaml
name: mlrun-project-workflow
on: [issue_comment]

jobs:
  submit-project:
    if: github.event.issue.pull_request != null && startsWith(github.event.comment.body, '/run')
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.7
      uses: actions/setup-python@v4
      with:
        python-version: '3.7'
        architecture: 'x64'
    
    - name: Install mlrun
      run: python -m pip install pip install mlrun
    - name: Submit project
      run: python -m mlrun project ./ --watch --run main ${CMD:5}
      env:
        V3IO_USERNAME: ${{ secrets.V3IO_USERNAME }}
        V3IO_API: ${{ secrets.V3IO_API }}
        V3IO_ACCESS_KEY: ${{ secrets.V3IO_ACCESS_KEY }}
        MLRUN_DBPATH: ${{ secrets.MLRUN_DBPATH }}
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }} 
        SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
        CMD: ${{ github.event.comment.body}}
```

### Secrets
Docs: [Working with secrets](https://docs.mlrun.org/en/latest/secrets.html)

```python
# Add secrets to project
project.set_secrets(secrets={'AWS_KEY': '111222333'}, provider="kubernetes")

# Run job with all secrets (automatically injects all project secrets for non-local runtimes)
project.run_function(fn)

# Retrieve secret within job
context.get_secret("AWS_KEY")
```

## MLRun Functions

### Essential Runtimes
Docs: [Kinds of functions (runtimes)](https://docs.mlrun.org/en/latest/concepts/functions-overview.html)

#### Job
```python
# Job - run once to completion
job = project.set_function(name="my-job", func="my_job.py", kind="job", image="mlrun/mlrun", handler="handler")
project.run_function(job)
```

#### Nuclio
```python
# Nuclio - generic real-time function to do something when triggered
nuclio = project.set_function(name="my-nuclio", func="my_nuclio.py", kind="nuclio", image="mlrun/mlrun", handler="handler")
project.deploy_function(nuclio)
```

#### Serving
```python
# Serving - specialized Nuclio function specifically for model serving
serving = project.set_function(name="my-serving", func="my_serving.py", kind="serving", image="mlrun/mlrun", handler="handler")
serving.add_model(key="iris", model_path="https://s3.wasabisys.com/iguazio/models/iris/model.pkl", model_class="ClassifierModel")
project.deploy_function(serving)
```

### Distributed Runtimes
Docs: [Kinds of functions (runtimes)](https://docs.mlrun.org/en/latest/concepts/functions-overview.html)

#### MPIJob (Horovod)
```python
mpijob = mlrun.code_to_function(name="my-mpijob", filename="my_mpijob.py", kind="mpijob", image="mlrun/mlrun", handler="handler")
mpijob.spec.replicas = 3
mpijob.run()
```

#### Dask
```python
dask = mlrun.new_function(name="my-dask", kind="dask", image="mlrun/ml-models")
dask.spec.remote = True
dask.spec.replicas = 5
dask.spec.service_type = 'NodePort'
dask.with_limits(mem="6G")
dask.spec.nthreads = 5
dask.apply(mlrun.mount_v3io())
dask.client
```

#### Spark Operator
```python
import os
read_csv_filepath = os.path.join(os.path.abspath('.'), 'spark_read_csv.py')

spark = mlrun.new_function(kind='spark', command=read_csv_filepath, name='sparkreadcsv') 
spark.with_driver_limits(cpu="1300m")
spark.with_driver_requests(cpu=1, mem="512m") 
spark.with_executor_limits(cpu="1400m")
spark.with_executor_requests(cpu=1, mem="512m")
spark.with_igz_spark() 
spark.spec.replicas = 2 

spark.deploy() # build image
spark.run(artifact_path='/User') # run spark job
```

### Resource Management
Docs: [Managing job resources](https://docs.mlrun.org/en/latest/runtimes/configuring-job-resources.html)

#### Requests/Limits (MEM/CPU/GPU)
```python
# Requests - lower bound
fn.with_requests(mem="1G", cpu=1)

# Limits - upper bound
fn.with_limits(mem="2G", cpu=2, gpus=1)
```

#### Scaling + Auto-Scaling
```python
# Nuclio/serving scaling
fn.spec.replicas = 2
fn.spec.min_replicas = 1
fn.spec.min_replicas = 4
```

#### Mount Persistent Storage
```python
# Mount Iguazio V3IO
fn.apply(mlrun.mount_v3io())

# Mount PVC
fn.apply(mlrun.platforms.mount_pvc(pvc_name="data-claim", volume_name="data", volume_mount_path="/data"))
```

#### Pod Priority
```python
fn.with_priority_class(name="igz-workload-medium")
```

#### Node Selection
```python
fn.with_node_selection(node_selector={"app.iguazio.com/lifecycle" : "non-preemptible"})
```

### Serving/Nuclio Triggers
Docs: [Nuclio Triggers](https://github.com/nuclio/nuclio-jupyter/blob/development/nuclio/triggers.py)
```python
import nuclio
serve = mlrun.import_function('hub://v2_model_server')

# HTTP trigger
serve.with_http(workers=8, port=31010, worker_timeout=10)

# V3IO stream trigger
serve.add_v3io_stream_trigger(stream_path='v3io:///projects/myproj/stream1', name='stream', group='serving', seek_to='earliest', shards=1)

# Kafka stream trigger
serve.add_trigger(
    name="kafka",
    spec=nuclio.KafkaTrigger(brokers=["192.168.1.123:39092"], topics=["TOPIC"], partitions=4, consumer_group="serving", initial_offset="earliest")
)

# Cron trigger
serve.add_trigger("cron_interval", spec=nuclio.CronTrigger(interval="10s"))
serve.add_trigger("cron_schedule", spec=nuclio.CronTrigger(schedule="0 9 * * *"))
```

### Building Docker Images
Docs: [Build function image](https://docs.mlrun.org/en/latest/runtimes/image-build.html), [Images and their usage in MLRun](https://docs.mlrun.org/en/latest/runtimes/images.html#images-usage)

#### Manually Build Image
```python
project.set_function(
   "train_code.py", name="trainer", kind="job",
   image="mlrun/mlrun", handler="train_func", requirements=["pandas==1.3.5"]
)

project.build_function(
    "trainer",
    # Specify base image
    base_image="myrepo/base_image:latest",
    # Run arbitrary commands
    commands= [
        "pip install git+https://github.com/myusername/myrepo.git@mybranch",
        "mkdir -p /some/path && chmod 0777 /some/path",    
    ]
)
```

#### Automatically Build Image
```python
project.set_function(
   "train_code.py", name="trainer", kind="job",
   image="mlrun/mlrun", handler="train_func", requirements=["pandas==1.3.5"]
)

# auto_build will trigger building the image before running, 
# due to the additional requirements.
project.run_function("trainer", auto_build=True)
```

## Logging
Docs: [MLRun execution context](https://docs.mlrun.org/en/latest/concepts/mlrun-execution-context.html)

```python
context.logger.debug(message="Debugging info")
context.logger.info(message="Something happened")
context.logger.warning(message="Something might go wrong")
context.logger.error(message="Something went wrong")
```

## Experiment Tracking
Docs: [MLRun execution context](https://docs.mlrun.org/en/latest/concepts/mlrun-execution-context.html), [Automated experiment tracking](https://docs.mlrun.org/en/latest/concepts/auto-logging-mlops.html)

#### Manual Logging
```python
context.log_result(key="accuracy", value=0.934)
context.log_model(key="model", model_file="model.pkl")
context.log_dataset(key="model", df=df, format="csv", index=False)
```

#### Automatic Logging
```python
from mlrun.frameworks.sklearn import apply_mlrun

apply_mlrun(model=model, model_name="my_model", x_test=X_test, y_test=y_test)
model.fit(X_train, y_train)
```

## Model Monitoring + Drift Detection
Docs: [Model monitoring overview](https://docs.mlrun.org/en/latest/monitoring/model-monitoring-deployment.html), [Batch inference](https://docs.mlrun.org/en/latest/deployment/batch_inference.html) 

#### Real Time Drift Detection
```python
# Log model with training set
context.log_model("model", model_file="model.pkl", training_set=X_train)

# Enable tracking for model server
serving_fn = import_function('hub://v2_model_server', project=project_name).apply(auto_mount())
serving_fn.add_model("model", model_path="store://models/project-name/model:latest") # Model path comes from experiment tracking DB
serving_fn.set_tracking()

# Deploy model server
serving_fn.deploy()
```

#### Batch Drift Detection
```python
batch_inference = mlrun.import_function("hub://batch_inference")
batch_run = project.run_function(
    batch_inference,
    inputs={
        "dataset": prediction_set_path,
        "sample_set": training_set_path
    },
    params={
        "model": model_artifact.uri,
        "label_columns": "label",
        "perform_drift_analysis" : True
    }
)
```

## Feature Store

## Real-Time Pipelines

## Hyperparameter Tuning
