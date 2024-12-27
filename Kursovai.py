from itertools import combinations
import numpy as np
from tabulate import tabulate
from math import floor, ceil
from copy import deepcopy

def round_if_close(value, tol=1e-6):
    """
    Округляет значение, если оно достаточно близко к целому числу.
    """
    if abs(value - round(value)) <= tol:
        return round(value)
    return value

class Instr:
    @staticmethod
    def to_down_index(i):
        """
        Преобразует индекс в нижний индекс переменной.
        Например: 1 -> "₁", 2 -> "₂".
        """
        subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return str(i).translate(subscript_map)

class DataConversion:
    def __init__(self, count_sq, count_ineq, max_min_task='max'):
        """
        Конструктор для задачи линейного программирования.

        :param count_sq: Количество переменных.
        :param count_ineq: Количество ограничений.
        :param max_min_task: Тип целевой функции ('max' для максимизации или 'min' для минимизации).
        """
        self.count_sq = count_sq  # Количество переменных
        self.count_ineq = count_ineq  # Количество ограничений
        self.c_coefficients = []  # Коэффициенты целевой функции
        self.restriction_coefficients = []  # Коэффициенты ограничений
        self.restriction_signs = []  # Знаки неравенств для ограничений
        self.right_hand = []  # Правая часть ограничений
        self.max_min_task = max_min_task  # Тип задачи (максимизация или минимизация)

    def set_objective_coefficients(self, coeff):
        """
        Устанавливает коэффициенты целевой функции.

        :param coeff: Список коэффициентов целевой функции.
        """
        if len(coeff) != self.count_sq:
            raise ValueError("Количество коэффициентов целевой функции должно совпадать с количеством переменных.")
        self.c_coefficients = coeff

    def add_constraint(self, constraint_input):
        """
        Добавляет ограничение к задаче.

        :param constraint_input: Строка или список с коэффициентами и правой частью ограничения.
        """
        # Преобразуем входные данные в список, если передана строка
        if isinstance(constraint_input, str):
            parts = constraint_input.split()
        else:
            parts = constraint_input
        
        # Извлекаем коэффициенты, знак и правую часть ограничения
        coeff = list(map(float, parts[:-2]))  # Все элементы кроме последних двух
        sign = parts[-2]  # Предпоследний элемент — знак неравенства
        b_start_table = float(parts[-1])  # Последний элемент — правая часть

        # Проверяем соответствие количества коэффициентов числу переменных
        if len(coeff) != self.count_sq:
            raise ValueError("Количество коэффициентов ограничения должно совпадать с количеством переменных.")

        # Добавляем ограничение в соответствующие списки
        self.restriction_coefficients.append(coeff)
        self.restriction_signs.append(sign)
        self.right_hand.append(b_start_table)

    def generate_dual(self):
        """
        Генерирует двойственную задачу линейного программирования.

        :return: Объект двойственной задачи.
        """
        # Проверка, что исходная задача полностью задана
        if not self.c_coefficients or not self.restriction_coefficients:
            raise ValueError("Необходимо полностью задать прямую задачу перед генерацией двойственной.")
        # Приводим все ограничения к стандартному виду (с '<=')
        for i in range(self.count_ineq):
            if self.restriction_signs[i] == '>=':
                # Инвертируем коэффициенты и правую часть, чтобы привести знак к '<='
                self.restriction_coefficients[i] = [-coef for coef in self.restriction_coefficients[i]]
                self.right_hand[i] = -self.right_hand[i]
                self.restriction_signs[i] = '<='
        # Создаем двойственную задачу с измененными параметрами
        dual_task = DataConversion(
            count_sq=self.count_ineq,  # Количество переменных в двойственной задаче
            count_ineq=self.count_sq,  # Количество ограничений в двойственной задаче
            max_min_task='min' if self.max_min_task == 'max' else 'max'  # Инвертируем тип задачи
        )
        # Устанавливаем коэффициенты целевой функции двойственной задачи
        dual_task.set_objective_coefficients(self.right_hand)
        # Перебираем каждую переменную прямой задачи для формирования ограничений двойственной задачи
        for j in range(self.count_sq):
            # Формируем коэффициенты для j-го ограничения двойственной задачи
            constraint = [self.restriction_coefficients[i][j] for i in range(self.count_ineq)]
            # Определяем знак для ограничения в двойственной задаче
            if self.restriction_signs[i] == '<=':
                dual_sign = '>='  # Прямое <= становится двойственным >=
            elif self.restriction_signs[i] == '>=':
                dual_sign = '<='  # Прямое >= становится двойственным <=
            else:
                raise ValueError(f"Неизвестный знак ограничения: {self.restriction_signs[i]}")
            # Добавляем ограничение в двойственную задачу
            constraint_str = " ".join(map(str, constraint))
            dual_task.add_constraint(f"{constraint_str} {dual_sign} {self.c_coefficients[j]}")


        # Добавляем ограничения на неотрицательность
        for j in range(self.count_ineq):
            # Создаем единичный вектор (j-я переменная двойственной задачи)
            constraint = [1 if i == j else 0 for i in range(self.count_ineq)]
            # Знак ограничения
            sign = ">="
            # Добавляем ограничение в двойственную задачу
            dual_task.add_constraint(
                f"{' '.join(map(str, constraint))} {sign} 0"
            )

        # Обновляем количество переменных и ограничений для двойственной задачи
        dual_task.count_sq = self.count_ineq
        dual_task.count_ineq = self.count_sq + self.count_ineq

        return dual_task

class DualSimplex:
    def __init__(self, lp_problem):
        """
        Инициализирует объект класса DualSimplex для решения линейной задачи методом двойственного симплекса.

        :param lp_problem: Объект линейной задачи, содержащий параметры задачи.
        """
        self.lp_problem = lp_problem  # Основная линейная задача
        self.dual_basis = []  # Список индексов переменных, входящих в текущий базис
        self.dual = lp_problem.generate_dual()  # Двойственная задача
        self.table = []  # Симплекс-таблица
        self.pseudo = []  # Псевдоплан (A_0)
        self.deltas = []  # Значения Δ для текущей симплекс-таблицы
        self.thetas = []  # Значения θ для выбора разрешающего столбца
        # Все возможные сочетания базисов для двойственной задачи
        self.basises = []
        for x in combinations(
            list(range(self.dual.count_ineq)),  # Генерируем список индексов ограничений
            r=self.dual.count_sq  # Количество переменных в двойственной задаче
        ):
            self.basises.append(x)

    def initial_basis(self):
        """
        Находит и инициализирует допустимый начальный базис.
        Перебираются возможные базисы, пока не найдется тот, который удовлетворяет всем ограничениям.
        """
        while True:
            # Берем последний базис из списка
            self.dual_basis = list(self.basises[-1])
            dual_basis = self.dual_basis[:]

            print("Рассматриваемый базис:", dual_basis)

            # Формируем правую часть ограничений и матрицу коэффициентов для текущего базиса
            b = [self.dual.right_hand[i] for i in dual_basis]
            A = [self.dual.restriction_coefficients[i] for i in dual_basis]
            A = np.array(A)
            b = np.array(b)

            print("Матрица коэффициентов A:", A)
            print("Правая часть b:", b)

            try:
                # Решаем систему линейных уравнений A * x = b
                solution = np.linalg.solve(A, b)
                solution = [float(x) for x in solution]
                print("Найденное решение системы уравнений:", solution)
            except np.linalg.LinAlgError:
                # Если система не имеет решения, удаляем текущий базис и переходим к следующему
                print("Система неразрешима. Переходим к следующему базису.")
                self.basises.pop()
                continue

            # Проверяем выполнение всех ограничений
            is_valid_basis = True
            for i, constraint in enumerate(self.dual.restriction_coefficients):
                constraint_value = sum(float(constraint[j]) * solution[j] for j in range(len(solution)))
                is_valid = (constraint_value >= self.dual.right_hand[i]) == (self.dual.restriction_signs[i] == ">=")
                is_valid_basis &= is_valid

            if is_valid_basis:
                # Если найден допустимый базис, завершаем процесс
                print("Выбранный начальный базис:", self.dual_basis)
                break

            # Если все базисы исчерпаны, выводим сообщение и завершаем процесс
            if not self.basises:
                print("Нет подходящего базиса.")
                break

            # Удаляем неподходящий базис и продолжаем
            print("Текущий базис не подходит. Переходим к следующему.")
            self.basises.pop()

    def pseudoplan_of_the_direct_problem(self):
        """
        Вычисляет начальный псевдоплан (A_0) для прямой задачи.
        """
        # Формируем матрицу коэффициентов ограничений для текущего базиса
        constraints = [self.dual.restriction_coefficients[i] for i in self.dual_basis]
        constraints_matrix = np.column_stack(constraints)

        # Получаем коэффициенты целевой функции
        objective = self.dual.c_coefficients

        # Решаем систему уравнений для нахождения псевдоплана
        try:
            self.pseudo = np.linalg.solve(constraints_matrix, objective)
            print("Текущий псевдоплан A_0:", self.pseudo)
        except np.linalg.LinAlgError:
            raise ValueError("Система уравнений не имеет решения. Проверьте корректность данных.")

        return

    def start_table(self):
        """
        Создает начальную симплекс-таблицу для двойственной задачи.
        """
        # Создаем правую часть (b) для начальной таблицы
        b_start_table = self.pseudo[:]
        self.table.append(b_start_table)
        
        # Для каждой ограничения создаем строки таблицы
        for i in range(self.dual.count_ineq):
            # Получаем коэффициенты ограничений для текущей строки
            constraints = [self.dual.restriction_coefficients[j] for j in self.dual_basis]
            
            # Получаем коэффициент целевой функции для текущего ограничения
            objective = self.dual.restriction_coefficients[i]
            
            # Решаем систему уравнений и добавляем результат в таблицу
            solved_constraints = np.linalg.solve(np.column_stack(constraints), objective)
            self.table.append(list(solved_constraints))
        
        # Преобразуем таблицу в формат, где все столбцы собраны вместе
        self.table = np.column_stack(self.table)


    def simplex_iteration(self, pivot_row, pivot_col):
        """
        Выполняет одну итерацию двойственного симплекс-метода.

        :param pivot_row: Индекс строки с разрешающим элементом.
        :param pivot_col: Индекс столбца с разрешающим элементом.
        """
        # Сохранение текущей симплекс-таблицы и обновление базисной переменной
        current_table = self.table[:]
        self.dual_basis[pivot_row] = pivot_col - 1

        # Создание новой таблицы для вычислений
        updated_table = [row[:] for row in current_table]

        # Определение разрешающего элемента
        pivot_element = updated_table[pivot_row][pivot_col]
        print("Разрешающий элемент: ", pivot_element)

        # Нормализация строки с разрешающим элементом
        updated_table[pivot_row] = [value / pivot_element for value in updated_table[pivot_row]]

        # Обновление остальных строк таблицы
        for row_index in range(len(updated_table)):
            if row_index != pivot_row:
                row_factor = updated_table[row_index][pivot_col]
                updated_table[row_index] = [
                    updated_table[row_index][col_index] - row_factor * updated_table[pivot_row][col_index]
                    for col_index in range(len(updated_table[row_index]))
                ]

        # Обновление симплекс-таблицы в объекте
        self.table = np.array(updated_table)

        # Пересчёт псевдоплана
        self.pseudo = [float(row[0]) for row in self.table]
        print("Текущий псевдоплан: ", self.pseudo)

        # Возврат обновлённой симплекс-таблицы
        return self.table

    def find_row(self):
        """
        Определяет строку для выбора разрешающего элемента.
        :return: Индекс строки с минимальным b (псевдопланом) или None, если нет отрицательных значений.
        """
        tableau = self.table
        array_1 = [tableau[x][0] for x in range(len(tableau))]  # Значения b
        array_2 = [x for x in array_1 if x < 0]  # Отрицательные значения b
        if not array_2:  # Если array_2 пустой
            return None
        return array_1.index(min(array_2))

    def calculate_deltas(self):
        """
        Вычисляет значения Δ для текущей симплекс-таблицы.
        """
        tableau = np.array(self.table, dtype=float)
        b_start_table = np.array(self.pseudo, dtype=float)
        basis_indices = np.array(self.dual_basis)
        objective = self.lp_problem.c_coefficients
        while len(objective) < self.dual.count_ineq:
            objective.append(0)
        objective = np.array(objective, dtype=float)
        deltas = []
        for j in range(self.dual.count_ineq + 1):
            delta = 0
            for i in range(len(basis_indices)):
                delta += objective[basis_indices[i]] * tableau[i][j]
            if j == 0:
                deltas.append(delta)
                continue
            delta -= objective[j - 1]
            deltas.append(delta)
        deltas = np.array(deltas)
        self.deltas = deltas

    def calculate_thetas(self):
        """
        Вычисляет значения θ для текущей симплекс-таблицы.
        """
        pivot_row = self.find_row()
        deltas = self.deltas[:]
        thetas = []
        table = self.table[:]
        for i in range(len(table[pivot_row])):
            if i == 0:
                thetas.append(float("inf"))
                continue
            if table[pivot_row][i] < 0 and abs(float(table[pivot_row][i])) >= 0.000001:
                thetas.append(-(deltas[i] / table[pivot_row][i]))
            else:
                thetas.append(float("inf"))
        thetas = np.array(thetas)
        self.thetas = thetas


    def print_table(self):
        """
        Выводит текущую симплекс-таблицу в удобочитаемом виде.
        """
        headers = [" ", "b"] + [f"A{i}" for i in range(1, self.table.shape[1])]
        ci_row = ["c_i", "-"] + list(self.lp_problem.c_coefficients[:self.table.shape[1] - 1])
        rows = []
        for i, (basis, row) in enumerate(zip(self.dual_basis, self.table)):
            basis_var = f"A{basis + 1}"  # Базисные переменные
            rows.append([basis_var] + list(row))
        deltas_row = ["Δ"] + list(self.deltas)
        thetas_row = ["Θ"] + list(self.thetas)
        print("Текущая симплекс-таблица:")
        print(tabulate([ci_row] + rows + [deltas_row] + [thetas_row], headers=headers, tablefmt="grid"))
        
    def final_table(self):
        """
        Выводит текущую симплекс-таблицу в удобочитаемом виде и сохраняет значения базисных переменных.
        """
        headers = ["B*", "b"] + [f"A{i}" for i in range(1, self.table.shape[1])]
        ci_row = ["c_i", "-"] + list(self.lp_problem.c_coefficients[:self.table.shape[1] - 1])
        rows = []
        self.basis_variables = {}  # Словарь для хранения значений базисных переменных

        for i, (basis, row) in enumerate(zip(self.dual_basis, self.table)):
            basis_var = f"A{basis + 1}"  # Имя базисной переменной
            rows.append([basis_var] + list(row))
            self.basis_variables[basis_var] = row[0]  # Сохраняем значение b для базисной переменной

        deltas_row = ["Δ"] + list(self.deltas)
        thetas_row = ["Θ"] + list(self.thetas)
        print("Итоговая симплекс-таблица:")
        print(tabulate([ci_row] + rows + [deltas_row] + [thetas_row], headers=headers, tablefmt="grid"))
        
    def find_solution(self):
        """
        Основной цикл метода двойственного симплекса для нахождения оптимального решения.
        """
        self.initial_basis()
        self.pseudoplan_of_the_direct_problem()
        self.start_table()
        
        while not all(x > 0 for x in self.pseudo):  # Пока псевдоплан содержит отрицательные значения
            self.calculate_deltas()
            self.calculate_thetas()
            self.print_table()
            
            pivot_row = self.find_row()  # Определяем направляющую строку
            if pivot_row is None:  # Если нет отрицательных значений b
                print("Нет отрицательных значений в псевдоплане. Решение найдено.")
                break
            
            data = self.thetas[:]
            thetas = [float(x) for x in data]
            
            if all(theta == float("inf") for theta in thetas):  # Проверяем, если все θ равны inf
                print("Все значения θ равны inf. Завершаем с решением 0.0, 0.0.")
                return 0.0, [0.0] * self.lp_problem.count_sq
            
            pivot_col = thetas.index(min(thetas))  # Определяем направляющий столбец
            self.simplex_iteration(pivot_row, pivot_col)  # Выполняем итерацию метода
        
        self.calculate_deltas()  # Вычисляем окончательные Δ
        self.final_table()  # Выводим итоговую таблицу

        # Проверяем, что lp_problem имеет атрибут count_sq
        count_sq = getattr(self.lp_problem, "count_sq", None)
        if count_sq is None:
            raise AttributeError("lp_problem не имеет атрибута count_sq")

        # Получаем итоговые значения переменных
        all_variables = {f"A{i+1}": 0 for i in range(count_sq)}
        for var, value in self.basis_variables.items():
            if var in all_variables:  # Учитываем только переменные до count_sq
                all_variables[var] = value

        # Ограничиваем вывод значениями только первых count_sq переменных
        variable_values_dict = {
            key: round_if_close(all_variables[key]) for key in list(all_variables.keys())[:count_sq]
        }

        # Преобразуем в список значений в порядке A1, A2, ..., An
        variable_values_list = [variable_values_dict[f"A{i+1}"] for i in range(count_sq)]

        # Прогоняем значение целевой функции через round_if_close
        optimal_value = round_if_close(self.deltas[0])

        print("\nЗначения переменных (в виде списка):")
        print(variable_values_list)
        print(f"\nОптимальное значение целевой функции: {optimal_value}")

        return optimal_value, variable_values_list


class LinProg():
    def __init__(self, is_max, sz_aim, aim, sz_coeffs, table, signs, free_members, s_table):
        # Инициализация класса линейного программирования
        self.is_max = is_max  # Максимизация (True) или минимизация (False)
        self.sz_aim = sz_aim  # Размер целевой функции
        self.aim = aim  # Коэффициенты целевой функции
        self.sz_coeffs = sz_coeffs  # Количество ограничений
        self.table = table  # Таблица коэффициентов ограничений
        self.signs = signs  # Знаки ограничений (True - ">=", False - "<=")
        self.free_members = free_members  # Свободные члены ограничений
        self.s_table = s_table  # Стандартная форма таблицы для симплекс-метода

        # Если задача минимизации, меняем знак коэффициентов целевой функции
        if not self.is_max:
            self.aim = [-x for x in self.aim]

        # Приведение ограничений к стандартному виду
        for i in range(self.sz_coeffs):
            for j in range(self.sz_aim):
                self.table[i][j] = self.table[i][j] if self.signs[i] else -self.table[i][j]
            self.free_members[i] = self.free_members[i] if self.signs[i] else -self.free_members[i]

        # Установка границ переменных (по умолчанию: от 0 до бесконечности)
        self.limits = [(0, float("inf")) for _ in range(self.sz_aim)]

    def solve(self):
        # Стек для хранения промежуточных решений
        stack = []

        # Определяем тип задачи: максимизация или минимизация
        max_perem = 'max' if self.is_max else 'min'

        # Создаем объект задачи в формате для симплекс-метода
        lp_problem = DataConversion(self.sz_aim, self.sz_coeffs, max_perem)
        copy_aim = deepcopy(self.aim)  # Копируем целевую функцию для восстановления позже
        lp_problem.set_objective_coefficients(self.aim)
        copy_table = deepcopy(self.s_table)  # Копируем таблицу ограничений для восстановления

        # Добавляем ограничения в задачу
        for i in range(self.sz_coeffs):
            lp_problem.add_constraint(self.s_table[i])

        # Решаем задачу методом двойственного симплекса
        simplex = DualSimplex(lp_problem)
        result_value, result_variables = simplex.find_solution()

        # Восстанавливаем исходные данные
        self.aim = copy_aim
        self.s_table = copy_table

        # Сохраняем лучшее решение
        best_solves = [([result_variables[i] for i in range(self.sz_aim)], -result_value if not self.is_max else result_value)]
        solves = [[self.table, self.free_members]]

        # Инициализация переменной для отслеживания оптимума
        optimum = float("-inf") if self.is_max else float("inf")

        # Пока есть дробные значения в решениях, продолжаем поиск
        while any(any(not elem.is_integer() for elem in slv[0]) for slv in best_solves):
            not_ints = set()  # Индексы решений с недопустимыми значениями
            new_solves = []

            for i in range(len(best_solves)):
                cur = best_solves[i][0]
                # Проверяем, превышает ли текущее решение текущий оптимум
                if (self.is_max and best_solves[i][1] > optimum) or ((not self.is_max) and best_solves[i][1] < optimum):
                    for j in range(len(cur)):
                        if not cur[j].is_integer():  # Если значение переменной дробное
                            not_ints.add(i)
                            # Генерируем новые ограничения для ветвления
                            new_str_left = [1 if k == j else 0 for k in range(self.sz_aim)]
                            new_str_right = [1 if k == j else 0 for k in range(self.sz_aim)]
                            new_solves.append([solves[i][0] + [new_str_left], solves[i][1] + [floor(cur[j])]])
                            new_solves.append([solves[i][0] + [new_str_right], solves[i][1] + [-ceil(cur[j])]])
                else:
                    not_ints.add(i)

            # Удаляем решения с недопустимыми значениями
            best_solves = [best_solves[i] for i in range(len(best_solves)) if i not in not_ints]
            solves = [solves[i] for i in range(len(solves)) if i not in not_ints]
            self.sz_coeffs += 1

            # Рассматриваем новые ограничения для каждого ветвления
            for i in range(len(new_solves)):
                lp_problem = DataConversion(self.sz_aim, self.sz_coeffs, max_perem)
                copy_aim = deepcopy(self.aim)
                lp_problem.set_objective_coefficients(self.aim)
                self.s_table = []

                # Добавляем ограничения в зависимости от знака свободных членов
                for j in range(self.sz_coeffs):
                    if new_solves[i][1][j] < 0:
                        self.s_table.append((new_solves[i][0][j]) + ['>='] + [abs(new_solves[i][1][j])])
                    elif new_solves[i][1][j] >= 0:
                        self.s_table.append(new_solves[i][0][j] + ['<='] + [new_solves[i][1][j]])

                for row in range(len(self.s_table)):
                    lp_problem.add_constraint(self.s_table[row])

                # Решаем новую задачу симплекс-методом
                simplex = DualSimplex(lp_problem)
                result_value, result_variables = simplex.find_solution()

                # Сохраняем результаты
                self.aim = copy_aim
                cur = ([result_variables[i] for i in range(self.sz_aim)], -result_value if not self.is_max else result_value)
                if (self.is_max and cur[1] > optimum) or (self.is_max and cur[1] < optimum):
                    solves.append(new_solves[i])
                    best_solves.append(cur)

        # Выводим лучшее найденное решение
        answer = max(best_solves, key=lambda x: x[1])
        print(f"F = {answer[1]}")
        for i in range(len(answer[0])):
            print(f"x{Instr.to_down_index(i + 1)} = {answer[0][i]}")


def main():
    is_max = input("Задача на поиск максимума или минимума? (max/min): ").strip().lower()
    is_max = True if is_max == "max" else False

    coeff = list(map(float, input("Введите коэффициенты целевой функции (через пробел): ").split()))
    sz_aim = len(coeff)

    sz_coeffs = int(input("Введите количество ограничений: "))
    print("Введите ограничения в формате: коэффициенты >= значение, например: 1 2 3 <= 4")
    s_table = []
    table = []
    signs = []
    free_members = []

    for i in range(sz_coeffs):
        row = input(f"Ограничение {i + 1}: ").split()
        # Разделение строки на коэффициенты, знак и правую часть
        *coeffs, sign, b = row
        coeffs = list(map(float, coeffs))  # Преобразование коэффициентов в числа
        b = float(b)  # Преобразование правой части в число

        simplex_row = coeffs + [sign, b]  # Собираем строку для симплекс-таблицы

        s_table.append(simplex_row)  # Добавляем строку в таблицу

        # Заполнение таблицы и других данных
        table.append(coeffs)
        free_members.append(b)
        signs.append(sign == "<=")

    # Создание и решение задачи линейного программирования
    linprog_solver = LinProg(is_max, sz_aim, coeff, sz_coeffs, table, signs, free_members, s_table)
    linprog_solver.solve()

if __name__ == "__main__":
    main()