import configparser

def config_file_parser(config_path: str) -> tuple[dict[str, str], dict[str, str]]:

  config = configparser.ConfigParser()
  config.read(config_path)
  lstm_params = config['lstm_params']
  file_params = config['file_params']
  
  return lstm_params, file_params

if __name__ == "__main__":
  file_path = "Project/configs/masked_lstm_config.ini"
  
  lstm_params, _ = config_file_parser(file_path)
  for k,v in lstm_params.items():
    print(f"{k} : {v}")