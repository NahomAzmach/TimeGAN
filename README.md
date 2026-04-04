# TimeGAN: Synthetic Energy Data Generator

This project implements the **Time-series Generative Adversarial Networks (TimeGAN)** architecture to synthesize realistic energy time-series data while preserving complex temporal dynamics.

### Inspiration
This implementation was deeply inspired by the groundbreaking paper:
📄 **[Time-series Generative Adversarial Networks (NeurIPS 2019)](./NeurIPS-2019-time-series-generative-adversarial-networks-Paper.pdf)** 
I highly recommend reading the attached PDF to understand the theoretical foundation of the four-network architecture (Embedding, Recovery, Generator, Discriminator) and the Stepwise Supervised Loss that allows the model to capture conditional temporal dynamics.

### What makes this project unique?

While many TimeGAN implementations exist strictly as backend scripts running on abstract benchmark datasets, this project uniquely applies the framework to the critical domain of **building energy physics** using the BuildingsBench dataset. By focusing on distinct scenarios—such as a standard Baseline compared to a severe Load Surge—the generator learns not just standard statistics, but the physical reality and temporal correlations of real-world energy consumption spikes. 

In the future, I plan to integrate a Generative AI feature that allows users to simply describe or explain a specific scenario in plain text. The application will then synthesize corresponding data sequences that perfectly match the user's described conditions, creating an incredibly flexible and intuitive data generation tool.

### Why is this needed?

Real-world building energy data is often very sensitive, proprietary, or severely imbalanced. Acquiring enough clean data covering rare "edge cases" (such as extreme load surges from heatwaves, grid anomalies, or severe weather) to accurately train AI models is incredibly difficult. 

This generator solves the data scarcity problem. It allows for the unlimited synthesis of diverse, highly-realistic temporal dynamics safely and accurately—without violating privacy or waiting years for anomalous data to naturally occur.
