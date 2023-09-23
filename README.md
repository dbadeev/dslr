# Проект Datascience X Logistic Regression
Реализация классификатора на основе стратегии «один-против-всех» с помощью модели логистической регрессии,  без использования специальных библиотек; статистический анализ данных; реализация пакетного градиентного спуска, мини-пакетного градиентного спуска, стохастического градиентного спуска; визуализация данных различными методами

## Начало Работы

### Копирование
Для копирования файлов Проекта на локальный компьютер в папку *<your_dir_on_local_computer>* выполните:

```
    $ git clone git@github.com:dbadeev/dslr.git <your_dir_on_local_computer>
```

### Описание файлов
* *dslr_en.pdf* - текст задания
* *requirements.txt* - список библиотек, необходимый для работы 
* *datasets* - папка с исходными данными + файл _evaluate.py_ для оценки качества построенной модели  
* *config.sh* - скрипт для установки необходимых библиотек + формирователь документации в формате _html_
* *docs* - папка с файлами для генератора документации _Sphinx_
* *maths* - папка с библиотекой утилит, используемых в проекте
    + *functions.py* - статистические функции
    + *logistic_regression.py* - модуль, включающий класс Логистической регрессии и вспомогательные функции
* *messages* - папка с файлами модуля вывода информации о ходе выполнения программы (сообщения об ошибках, окончания рабы и т.п.)
* *histogram.py* - программа отрисовки гистограммы для поиска схожих распределений свойств набора данных по классам
* *scatter_plot.py* - программа поиска по матрице корреляции и отрисовки диаграммы рассеяния наиболее схожих свойств из набора данных
* *pair_plot.py* - программа отображения попарных диаграмм рассеяния для свойств в наборе данных
* *logreg_train.py* - программа обучения модели логистической регрессии
* *logreg_predict.py* - программа, предсказывающая наиболее вероятный класс по свойствам объекта
<br>

## Запуск программ
### Начало работы
0. Для работы с Проектом сначала рекомендуется создать и активировать виртуальное окружение, в которое будут устанавливаться необходимые библиотеки.
```
$ python3 -m venv venv
$ source venv/bin/activate
(venv)
```

### Необходимые библиотеки и документация
1. Для установки необходимых библиотек и формирования html документации по Проекту, из корневой папки проекта в командной строке нужно запустить скрипт _config.sh_. <br>
```
$ ./config.sh
```

Сформированные файлы документации расположены в папке _docs/build/html/_. <br> 
Для просмотра документации в браузере нужно открыть файл _docs/build/html/index.html_

### Статистика данных
2. Для получения информации по статистикам свойств объектов из папки с проектом в командной строки запускается программа _describe.py_. Для получения расширенной информации используется ключ _-e_.
```
$ python3 describe.py -e datasets/dataset_train.csv
```
Результат работы программы - информация по статистическим характеристикам каждого свойста объектов. <br>
Набор статистических характеристик по которым приводится отчет подробно описан в документации сформированной на предыдущем шаге.
### Визуализация данных
3. Для получения изображения гистограмм распределения свойств объектов для каждого из классов по данным из датасета, из папки с проектом в командной строки запускается программа _histogram.py_
```
$ python3 histogram.py
```
Результат работы программы - файл _histogram_all.png_ в корневой папке проекта. <br>

4. Для поиска по матрице корреляции и получения изображения диаграмм рассеяния наиболее схожих свойств в наборе данных, из папки с проектом в командной строки запускается программа _scatter_plot.py_
```
$ python3 scatter_plot.py
```
Результат работы программы - файл _scatter_plot.png_ в корневой папке проекта. <br>

5. Для получения изображения попарных диаграмм рассеяния свойств в наборе данных, из папки с проектом в командной строки запускается программа _pair_plot.py_
```
$ python3 pair_plot.py
```
Результат работы программы - файл _pair_plot.png_ в корневой папке проекта. <br>

### Обучение, запуск и оценка качества модели
5. Для получения информации о параметрах программы обучения модели, из корневой папки проекта нужно выполнить программу *logreg_train.py* с ключом _-h_:
```
$ python3 logreg_train.py -h          
usage: logreg_train.py [-h] [--gradient GRAD] [--debug] data

positional arguments:
  data                  Path to train data file

optional arguments:
  -h, --help            show this help message and exit
  --gradient GRAD, -g GRAD
                        Gradient descent method: "batch" (default), "mini_batch", "sgd"
```

6. После успешной отработки программы обучения модели в папке _datasets_/ будет сформирован файл _weights.csv_, в котором будут записаны найденные веса (параметры) по классам для каждого свойства, отобранного для построения модели. <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>

7. Для получения информации о параметрах программы-классификатора, предсказывающей на основе стратегии «один-против-всех» класс, к которому относится объект по его свойствам, из корневой папки проекта нужно выполнить программу *logreg_predict.py* с ключом _-h_:
```
$ python3 logreg_predict.py -h                      
usage: logreg_predict.py [-h] data weights

positional arguments:
  data        Path to data file
  weights     Path to weights file

optional arguments:
  -h, --help  show this help message and exit
```


8. После успешной отработки программы-классификатора в папке *datasets*/ будет сформирован файл _houses.csv_ с результатами предсказаний по каждому объекту из файла с тестовыми данными и на экран выведена информация об окончании работы.  <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>

9. Для оценки качества модели нужно перейти из корневой папки проекта в папку _datasets_/, где находятся файлы _houses.csv_, _dataset_truth.csv_, а также программа _evaluate.py_ и запустить последнюю:
```
$ cd datasets 

$ python3 evaluate.py
```
На экран будет выведена информация о количестве объектов в каждом из _csv_ файлов и доля верно предсказанных результатов.

<br>

## Подробности

Подробнее о Проекте, анализ данных, примеры использования - по ссылке Wiki (https://github.com/dbadeev/dslr/wiki).

<br>

## Авторы

*loram (Дмитрий Бадеев)* - описание и реализация модели логистической регрессии <br>
*gdorcas (Татьяна Смирнова)* - реализация статистических функций, программ визуализации, стат анализ, документация с использованием Sphinx

<br><br>

## Результат в School 21

