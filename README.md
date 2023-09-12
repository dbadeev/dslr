# Проект Datascience X Logistic Regression
Реализация модели с несколькими классификаторами с использованием логистической регрессии «один против всех» без использования специальных библиотек; статистический анализ данных; реализация пакетного градиентного спуска, мини-пакетного градиентного спуска, стохастического градиентного спуска; визуализация данных различными методами

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
* *scatter_plot.py* - программа отрисовки диаграммы рассеяния для поиска наиболее схожих свойств набора данных
* *pair_plot.py* - программа отображения попарных отношений для свойств набора данных
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
1. Для установки необходимых библиотек и формирования html документации по Проекту, из папки с проектом в командной строки запускается скрипт _config.sh_. <br>
```
$ ./config.sh
```

Сформированные файлы документации расположены в папке _docs/build/html/_. <br> 
Для просмотра документации в браузере нужно открыть файл _docs/build/html/index.html_

### Визуализация данных
2. Чтобы вывести на экран информацию о параметрах, 
из папки *<your_dir_on_local_computer>* выполните *train.py* с ключом _-h_:

```
$ python3 train.py -h          
usage: train.py [-h] [--path PATH] [--loss_control LOSS_CONTROL] [--epochs EPOCHS] [--learning_rate ETA] [--loss_graphics]
                [--predict_data] [--animation] [--debug] [--quality]

options:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  Path to data file (data.csv by default)
  --loss_control LOSS_CONTROL, -l LOSS_CONTROL
                        Epoch iterations will stop while gets loss_control value(1e-12 by default)
  --epochs EPOCHS, -e EPOCHS
                        Set the epochs number (1500 by default)
  --learning_rate ETA, -a ETA
                        Set the learning rate eta (0.2 by default)
  --loss_graphics, -g   Diagram with loss function depends on epochs
  --predict_data, -t    Diagram with data values and line prediction
  --animation, -c       Animation with prediction evolution while training
  --debug, -d           Print info about each stage of program
  --quality, -q         Model quality (R-square, MSE)

```

2. После успешной отработки программы обучения модели будет сформирован файл _coefs.csv_, в котором будут записаны найденные коэффициенты формулы вычисления предсказания цены автомобиля по заданному пробегу. <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>
3. Для получения информации о параметрах программы, предсказывающей примерную цену автомобиля в зависимости от пробега, из папки *<your_dir_on_local_computer>* выполните *predict.py* с ключом _-h_:
```
$ python3 predict.py -h 
usage: predict.py [-h] [--debug] [--mileage MILEAGE]

options:
  -h, --help            show this help message and exit
  --debug, -d           Print info about each stage of program
  --mileage MILEAGE, -m MILEAGE
                        Car mileage for price prediction (non-negative int)
```
<br>
4. После успешной отработки программы предсказания, на экран будет выведена информация о примерной стоимости автомобиля с указанным пробегом.  <br>
В случае ошибки, будет выдано соответствующее сообщение. <br><br>

## Подробности

Подробнее о Проекте, примеры использования - по ссылке Wiki (https://github.com/dbadeev/ft_linear_regression/wiki).

<br>

## Авторы

*loram (Дмитрий Бадеев)* - описание и реализация модели логистической регрессии <br>
*gdorcas (Татьяна Смирнова)* - реализация статистических функций, стат анализ, документация с использованием Sphinx

<br><br>

## Результат в School 21

