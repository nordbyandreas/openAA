
# "pip install pyfiglet" hvis du ikke har det
from pyfiglet import Figlet
from input_parser import InputParser


class OpenAA():

    def set_case_manager(self, case_manager):
        self.case_manager = case_manager

    def get_case_manager(self):
        return self.case_manager

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model






def main():

    f = Figlet(font='slant')
    print(f.renderText('        '))
    print(f.renderText(' openAA:'))
    print("         'h' for help, 'q' to quit  \n")
    print(f.renderText('         '))
    openAA = OpenAA()
    inputparser = InputParser(openAA)

    while True:
        cmd = input('$$$: ')
      
        if cmd == "q":
            break
        elif cmd == "h":
            print("\n -List of commands: -----------\n")
            print("1. load_data   (load datasets)")
            print("2. load_json   (load settings from json file)")
            print("3. setup_model (set GANN and training parameters)")
            print("4. visualize   (set visualization modes)")
            print("5. run_model   (build the GANN and train it)")
            print("6. view_model  (view GANN and training paramters)")
            print("7. predict     (run some cases through the trained GANN and look at the predictions)")
            print("\n For now you will have to read through the code to see how these are used.\n Might add DOCS and better help-instructions here later")
        elif cmd == "":
            continue
        else:
            inputparser.evaluator(cmd)



if __name__ == "__main__":
    main()