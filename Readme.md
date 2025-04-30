# Network host profiling using Federated Learning with Local Autoencoder

The goal of the experiments is to apply federative learning to create a global model while locally the models are created using Autoencoder that profiles the normal host traffic.

Local profile is created as unsupervised anomaly detection models by training an autoencoder on “normal” communication data from  host communciation and then using the reconstruction error to flag anomalies.

Federated Learning is a decentralized machine learning approach where multiple participants collaboratively train a shared model while keeping their raw data localized. Instead of transmitting sensitive data to a central server, each participant trains the model (autoencoder model) on their own data and only shares model updates (like gradients or weights) with the central aggregator. 

## Approach

1. Collect and Clean Data:
Collect advanced flow information (e.g., TLS parameters, packet sizes, record sizes) and perform any necessary cleaning and normalization. Ensure that your training data set represents typical, non-anomalous behavior. Processing of the source capture file into JSON data is done using the `Shar.Export-TlsConnections.ps1` script. See the *Tools* section for how to use this tool.

2. Feature Engineering:
Convert categorical parameters (e.g., TLS cipher suites) to numerical representations (e.g., one-hot encoding) and scale numerical features. For sequence data (like packet size sequences), consider techniques such as padding or time-window aggregation.

3. Autoencoder Architecture: The autoencoder architecture may vary depending on the input data:

    * Static Data: each connection or flow is represented by with a fixed-size feature vector, a standard feed-forward autoencoder (with fully connected layers) may work well.

    * Sequential Data: for connection representation as a varying-length record size sequences, ecurrent autoencoders (e.g., LSTM or GRU-based) can capture temporal patterns.

3. Training the Autoencoder

    * Train on Normal Behavior: Feed the autoencoder only normal data so that it learns to compress and then reconstruct these typical patterns. The idea is that the autoencoder will “memorize” the usual behavior of the host.

    * Loss Function:
Use a reconstruction loss (commonly mean squared error) to measure how well the input is being reconstructed.

4. Defining Anomaly Thresholds

    * Reconstruction Error: Once trained, calculate the reconstruction error for each connection/flow. Under normal conditions, this error should be relatively low.

    * Set a Threshold: Analyze the distribution of reconstruction errors on your validation (or training) set and choose a threshold above which an instance is considered anomalous. You may need to tune this threshold to balance false positives and false negatives.


## Tools

Creating JSON from the source capture files requires installation of the [tshark tool](https://tshark.dev/setup/install/) and the [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell?view=powershell-7.5) environment. The script to decode the capture file and extract information for each TLS connection and output it as JSON is `Shark.Export-TlsConnections.ps1`, available in the `scripts` subfolder.

```bash
 ./scripts/Shark.Export-TlsConnections.ps1 -PcapFolder PATH-TO-CAPUTRE-FOLDER -Recurse $true -OutPath PATH-TO-OUTPUT-JSON-FOLDER 
```

The following example will process all the capture files for Windows applications' communication:

```bash
./scripts/Shark.Export-TlsConnections.ps1 -PcapFolder ../Datasets/Windows/Captures  -Recurse $true -OutPath ./datasets/windows.tls/
```

## Available Datasets

| Dataset | Description|
| --- | --- | 
[CCCS-CIC-AndMal-2020](https://www.unb.ca/cic/datasets/andmal2020.html) |  A comprehensive and huge android malware dataset, named CCCS-CIC-AndMal-2020. The dataset includes 200K benign and 200K malware samples totalling to 400K android apps with 14 prominent malware categories and 191 eminent malware families.|
[CIC-AAGM2017](https://www.unb.ca/cic/datasets/android-adware.html) | CICAAGM dataset is captured by installing the Android apps on the real smartphones semi-automated. The CICAAGM dataset consists of the network traffic of both the malware and benign (20% malware and 80% benign) |
