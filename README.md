# RcPacing
Percentile Risk-Constrained Budget Pacing for Guaranteed Display Advertising in Online Optimization

# RCPacing Offline Evaluation

## DataSet

### Format

- first row: budget of each campaign, format:
  - budgetPv|campaignId:budget;campaignId:budget
- other rows: request logs, format:
  - hh:mi|campaign:CTR;campaign:CTR

### Notice
- CTR 
  - Due to the requirement for data confidentiality, the CTR has been linearly amplified by a certain factor and saved as an integer.
- request
  - request are sampled from real-world data.


### Realized Models
- DMD
- PDOA
- AUAF
- Smart Pacing
- **RCPacing** (our method)


## Run

- Env 
  - python 3.8+
  - numpy
  - scipy

- Cmd
  - `python main.py --model rcpacing --data_path ./data/supply_demand_data.txt`
