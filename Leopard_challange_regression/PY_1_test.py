class OperationsForStudentsTxt:

    def __init__(self):
        with open('students.txt', 'r', encoding='utf-8') as file:
            self.list_of_names = [i.strip() for i in file.readlines()]

        #print(self.list_of_names)

    def write_result_list_of_students(self):
        with open('1.txt', 'w', encoding='utf-8') as file:
            for i in self.list_of_names:
                file.write(i+"\n")

    def add_new_stud(self, surname, name):
        self.list_of_names.append(f"{name} {surname}")
        self.list_of_names = sorted(self.list_of_names)
        self.write_result_list_of_students()

    def fild_stud(self, surname, name=""):
        if name == "":
            find_stud = [i for i in self.list_of_names if surname in i]
            if len(find_stud) == 0:
                print(f"Студентов с фамилией {surname} не найдено")
                return -1
            else:
                print(" ".join(find_stud))

    def change_stud(self, surname, name, new_surname="", new_name=""):
        self.del_stud(surname, name)
        if new_surname == "": new_surname = surname
        if new_name == "": new_name = name
        self.add_new_stud(new_surname, new_name)

    def del_stud(self, surname, name=""):
        if name == "":
            find_stud = [i for i in self.list_of_names if surname in i]
            if len(find_stud) == 0:
                print(f"Студентов с фамилией {surname} не найдено")
                return -1
            elif len(find_stud) == 1:
                name = find_stud.pop().replace(surname, "").strip()
            else:
                print("Введите имя студента для удаления: ")
                print("    ".join(find_stud))
                name = input().strip()

        if f"{name} {surname}" in self.list_of_names:
            self.list_of_names.remove(f"{name} {surname}")

        self.write_result_list_of_students()


b = OperationsForStudentsTxt()

print("Конец")