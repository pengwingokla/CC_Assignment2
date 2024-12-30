## Project Description
The SparkWinePredictor project focuses on building a parallel machine learning application to predict wine quality using Apache Spark's MLlib on Amazon AWS. This project involves training, validating, and testing a wine quality prediction model across multiple EC2 instances and deploying the model using Docker for simplified distribution and execution.

The project objectives are as follows:

* Parallel ML Model Training: Utilize Spark's distributed computing capabilities to train the model in parallel on 4 AWS EC2 instances using the provided TrainingDataset.csv.
* Model Validation and Optimization: Use the ValidationDataset.csv to validate and fine-tune the model, ensuring optimal performance.
* Model Testing: Evaluate the trained model's performance on unseen data using the F1 score as a key performance metric.
* Dockerized Deployment: Package the Spark application and trained model into a Docker container to enable seamless deployment on a single EC2 instance for prediction.
  
The project will employ Spark's MLlib to implement a simple linear regression or logistic regression model for classification, starting with basic models and exploring additional ML algorithms to enhance performance. The application will classify wine quality scores (1 to 10) based on the provided datasets.

This hands-on project showcases the power of Apache Spark for distributed machine learning, the versatility of MLlib for regression and classification, and the scalability of AWS cloud infrastructure for high-performance computing tasks.

## Project Action Items
0. Launch 4 EC2 instances on AWS to parallelize model training.
0. SSH into the EC2 Instance
0. Transfer Files to the Instance

### (Option 1) Running the Project Without Docker on the Instance
1. Install Java OpenJDK (required for Spark) and Apache Spark on all instances
2. Configure the instances to run Spark on Ubuntu Linux.
3. Submit the Spark Job

### (Option 2) Running the Project with Docker on the Instance
1. Install Docker on the Instance
2. Build the Docker Image on the Instance
3. Run the Docker Container
4. Submit the Spark Job

## Transfer Dataset to master instance
`scp -i <your-key>.pem TrainingDataset.csv ubuntu@<master-public-ip>:/home/ubuntu/`
<br> `scp -i <your-key>.pem ValidationDataset.csv ubuntu@<master-public-ip>:/home/ubuntu/`

## Set Up AWS EC2 Instances
#### Install required software on EC2 instances
SSH to connect to each instance, install Java and check: 
<br> `ssh -i <your-key>.pem ubuntu@<instance-public-ip>`
<br> `sudo apt update && sudo apt upgrade -y`
- Install Java
<br> `sudo apt install openjdk-11-jdk -y`
<br> `java -version`
- Install Spark
<br> `wget https://downloads.apache.org/spark/spark-<version>/spark-<version>-bin-hadoop3.tgz`
- Extract and move Spark to /opt: 
<br> `tar -xzf spark-<version>-bin-hadoop3.tgz`
<br> `sudo mv spark-<version>-bin-hadoop3 /opt/spark`
- Add Spark to your PATH by editing `~/.bashrc`:
<br> `echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc`
<br> `echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc`
<br> `source ~/.bashrc`
- Install Scala (required for Spark)
<br> `sudo apt install scala -y` && `scala -version`

## Set up passwordless SSH Access
Configure passwordless SSH access from the master node to each worker node to streamline the copying process.
<br> **1) Generate an SSH Key Pair on the Master Node:**
<br> `ssh ubuntu@<MASTER_NODE_IP>`
<br> `ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -q -N ""`
<br> This creates a key pair:
* Private key: `~/.ssh/id_rsa`
* Public key : `~/.ssh/id_rsa.pub`

<br> Confirm the key exists on master node: `ls ~/.ssh/id_rsa.pub`

<br> **2) Manually Copy the Public Key to All Worker Nodes**
<br> On the master node, output the public key: 
<br> `cat ~/.ssh/id_rsa.pub`
<br> For each worker node, copy the public key from the master node to the worker's `~/.ssh/authorized_keys`. Repeat this step for all worker nodes:
<br> `echo "<PASTED_PUBLIC_KEY>" >> ~/.ssh/authorized_keys`
<br> Ensure proper permissions on the worker node:
<br> `chmod 600 ~/.ssh/authorized_keys`
<br> `chmod 700 ~/.ssh`

<br> From the master node, test logging into each worker node without a password:
<br> `ssh ubuntu@<WORKER_NODE_IP>`

## Set Up Spark Cluster
The parallel training implementation in this project leverages Apache Spark's distributed computing capabilities.
This project sets up Apache Spark standalone cluster with:
* 1 Master Node: Coordinates and schedules the execution of tasks.
* 3 Worker Nodes: Execute tasks in parallel, processing data distributed across them.
Each worker node has a specified number of cores and memory assigned.

#### Commands
<br>Start Spark master and worker instances
<br>`$SPARK_HOME/sbin/start-master.sh`
<br>`$SPARK_HOME/sbin/start-slave.sh spark://172.31.25.1:7077`

<br> Stop existing worker process: `$SPARK_HOME/sbin/stop-worker.sh`
<br>Check for Spark worker processes:
`ps -ef | grep Worker`

#### Spark Web UI: 
`http://<master-public-ip>:8080`

### Configure Spark Cluster
<br> 1) Verify Spark Installation: `/opt/spark/bin/spark-shell --version`

<br> 2) Set environment variables (add to ~/.bashrc):
<br>`export SPARK_HOME=/opt/spark`
<br>`export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin`
<br>`export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")`
<br> Apply changes: `source ~/.bashrc`

<br> 3) Configure `spark-env.sh`
<br> `SPARK_MASTER_HOST=<master_node_private_ip>`
<br> `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
<br> `HADOOP_CONF_DIR=$SPARK_HOME/conf`
<br> `SPARK_WORKER_CORES=6`     # Adjust based on your instance resources
<br> `SPARK_WORKER_MEMORY=18G`  # Adjust based on your instance resources
<br> `SPARK_WORKER_INSTANCES=1`
<br> `SPARK_EXECUTOR_INSTANCES=1`

<br> 4) Distribute the spark-env.sh file to all worker nodes:
<br> `scp $SPARK_HOME/conf/spark-env.sh ubuntu@<worker_node_ip>:$SPARK_HOME/conf/spark-env.sh`

<br> 5) Configure worker files
<br> `sudo nano $SPARK_HOME/conf/slaves`
<br> Add the private IPs or hostnames of all worker nodes, one per line:
<br> <worker_node_1_private_ip>
<br> <worker_node_2_private_ip>
<br> <worker_node_3_private_ip>
<br> <worker_node_4_private_ip>

<br> Distribute the worker file to all worker nodes:
<br> `scp $SPARK_HOME/conf/slaves ubuntu@<worker_node_ip>:$SPARK_HOME/conf/slaves`

<br> 6) Start Spark Cluster
<br> On master node: `$SPARK_HOME/sbin/start-master.sh`
<br> On worker node: `$SPARK_HOME/sbin/start-slave.sh spark://<master_node_private_ip>:7077`

## Use Docker in a Spark Cluster
### Set up Docker

<br> 0. Transfer Files to the Instance: Use `scp` to transfer all necessary files (e.g., Dockerfile, app.py, requirements.txt, TrainingDataset.csv) to the EC2 instance.

<br> 1. Build and Tag Your Docker Image Locally
<br> `docker build -t wine-quality-app .`
<br> verify using: `docker images`

<br> 2. Push the Docker Image to a Registry - Docker Hub
<br> Login to Docker Hub: `docker login`
<br> Tag and push images: `docker tag wine-quality-app DOCKERHUB-USERNAME/wine-quality-app:latest
docker push DOCKERHUB-USERNAME/wine-quality-app:latest`

<br> 3. Install Docker on All Nodes
<br> `ssh -i "your_key.pem" ubuntu@node-ip`
<br> `sudo apt update`
<br> `sudo apt install -y docker.io`
<br> `sudo systemctl start docker`
<br> `sudo systemctl enable docker`

<br> 4. Configure Docker on EC2 Instance
- Pull the Docker Image on All Nodes
<br> `sudo docker pull DOCKERHUB-USERNAME/wine-quality-app:latest`
- Add user to the Docker group to avoid needing sudo for Docker commands: 
<br> Check if your user (ubuntu) is part of the docker group:
<br> `groups`
<br> If not, add by using:
<br> `sudo usermod -aG docker $USER`
<br> `newgrp docker`
<br> `docker run hello-world`

### Running the Project with Docker
<br> 5. Build the Docker Image on the Instance
<br> `cd /home/ubuntu/code`
<br> `docker build -t wine-quality-app .`

<br> 6. Run the Docker Container
<br> `docker run -it --rm \
--network="host" \
wine-quality-app`

<br> 7. Configure Spark to Use Docker
<br> Edit the spark-env.sh file:
<br> `sudo nano $SPARK_HOME/conf/spark-env.sh`
<br> Add the following line:
<br> `SPARK_EXECUTOR_OPTS="--conf spark.executor.docker.image=your-image-name"`
`SPARK_DRIVER_OPTS="--conf spark.driver.docker.image=your-image-name"`

<br> 8. Submit the Spark Application

### Docker Hub:
https://hub.docker.com/repository/docker/chloecodes/wine-quality-app/general
## Training and Prediction Application
Spark will use the Docker container to execute the tasks in parallel across all nodes.

<br> Submit the Spark Application on master node:
<br> `$SPARK_HOME/bin/spark-submit \`
<br>`  --master spark://172.31.25.1:7077 \`
<br>`  --deploy-mode client \`
<br>`  --executor-memory 18G \`
<br>`  --total-executor-cores 6 \`
<br>`  /home/ubuntu/code/app.py`

## Key Notes
- With Docker:
Simplifies setup and ensures consistency across all environments.
Requires a properly built Docker image.
- Without Docker:
Requires manual setup on each node, including installing dependencies and configuring Spark.

<br>
<br>
<br>
<br>

