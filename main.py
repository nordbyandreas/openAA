
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

        if cmd == "h":
            print("help is coming..!")
        
        inputparser.evaluator(cmd)



if __name__ == "__main__":
    main()