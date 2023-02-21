from trainer.trainer import Trainer
from utils.parser import get_parser

def main():
    parser = get_parser()
    config = parser.parse_args()
    trainer = Trainer(config)
    
    trainer.eval_partial()
    
if __name__ == "__main__":
    main()