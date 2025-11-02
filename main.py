import argparse
from api.server import create_app
from examples.basic_usage import basic_demo
from examples.advanced_reasoning import advanced_demo

def main():
    parser = argparse.ArgumentParser(description="Multimodal Reasoning Engine")
    parser.add_argument('--mode', type=str, choices=['api', 'demo', 'advanced'], default='demo')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        app = create_app()
        print(f"Starting API server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    
    elif args.mode == 'demo':
        basic_demo()
    
    elif args.mode == 'advanced':
        advanced_demo()

if __name__ == "__main__":
    main()