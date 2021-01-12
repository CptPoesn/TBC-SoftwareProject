import rasa.cli.shell
import rasa.cli.run

class Namespace:
	def __init__(self):
		self.model = "model"
args=Namespace()
args.model="C:/Users/schmi/Softwareprojekt/rasa_test"
args.connector = "cmdline"
args.endpoints = ""
args.credentials = ""
args.enable_api = False
args.remote_storage = "."
rasa.cli.run.run(args)
rasa.cli.shell.shell(args)

