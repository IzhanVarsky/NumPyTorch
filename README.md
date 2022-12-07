# NumPyTorch - numpy-only pytorch-like framework

Небольшой фреймворк, основанный на numpy, максимально приближенный к реализации PyTorch + тесты к слоя

### Реализованные классы:

* [nn.Variable](nn/variable.py) - класс для хранения вектора оптимизируемых значений, хранит в том числе накопленный
  градиент

Классы-аналоги из `torch.nn.*` с реализованными методами `forward(input)` и `backward(grad)`:

* [nn.Module](nn/module.py)
* [nn.BatchNorm1d/nn.BatchNorm2d](nn/batchnorm.py)
* [nn.Sequential](nn/container.py)
* [nn.Conv2d](nn/conv.py)
* [nn.Dropout](nn/dropout.py)
* [nn.Flatten](nn/flatten.py)
* [nn.Linear](nn/linear.py)
* [nn.MaxPool2d](nn/pooling.py)
* [nn.MSELoss](nn/loss.py)
* [nn.CrossEntropyLoss](nn/loss.py) - в torch этот класс является на самом деле 'CrossEntropy with Softmax', однако в
  данном случае была реализована только кросс-энтропия без softmax (эту активацию нужно добавлять отдельно)
* [nn.ReLU](nn/activation.py)
* [nn.Softmax](nn/activation.py)
* [nn.Sigmoid](nn/activation.py)
* [nn.Tanh](nn/activation.py)

Класс [nn.Net](nn/net.py), агрегирующий саму нейронную сеть и функцию потерь, которую необходимо применить к входу +
выходу этой сети.

Оптимизаторы, частично реализующие возможности оптимизаторов из `torch.optim.*`:

* [optim.SGD](optim/sgd.py)
* [optim.Momentum](optim/momentum.py)
* [optim.RMSProp](optim/rmsprop.py)
* [optim.Adam](optim/adam.py)
* [optim.RAdam](optim/radam.py)

Планировщики скорости обучения:

* [optim.ConstantLR](optim/lr_scheduler.py)
* [optim.ExpoLR](optim/lr_scheduler.py)
* [optim.LinearLR](optim/lr_scheduler.py)
* [optim.TimeBasedLR](optim/lr_scheduler.py)

В обучающих целях были оставлены следующие медленные реализации сверточных классов:

* [nn.SlowMaxPool2d](nn/pooling.py)
* [nn.SlowConv2d](nn/conv.py)

Для всех реализованных слоев были написаны тесты, проверяющие как корректность работы `forward`, так и `backward`,
использующие для сравнения промежуточные результаты классов-аналогов из PyTorch.

### Объяснения forward и backward:

#### База

Нейронную сеть можно представить в голове как одну большую функцию, в которую мы подаем какой-то вход `x` и в ответ
после каких-то внутренних преобразований получаем выход `predicted`.
Весь процесс преобразования нашего входа в выход - это функция `forward(x)`.

Поскольку у нас должно происходить обучение, то наша функция `forward` внутри должна содержать/использовать параметры
(веса), которые мы будем оптимизировать, чтобы выход был более близок к тому выходу, который мы на самом деле ожидаем
(ground truth).

Для того чтобы понять, насколько необходимо изменить внутренние веса нейронной сети, нужно посчитать градиенты функции
потерь по весам нейронной сети (with respect to the weights).
Используя эти градиенты, можно оптимизировать веса согласно формуле обновления весов при градиентном спуске:
`W_new = W_old - lr * grad_W_old`.

Как именно посчитать эти градиенты?
Представим нашу функцию `forward` таким образом: `f_k(f_k-1(...f_3(f_2(f_1(x)))))`.
После выхода `f_k` мы применим последнюю функцию - функцию потерь `Loss` - и относительно этой функции мы должны считать
градиенты, т.к. мы хотим именно эту функцию (функцию потерь) оптимизировать.
Т.е. нам нужно найти `dLoss/dW` - градиент функции `Loss` по оптимизируемому параметру `W`.

Вспомним цепное правило: `dLoss/dW = dLoss/dfx * dfx/dW`.
В нашем случае его можно применить так: `dLoss/dW = dLoss/df_k * df_k/dW`.
Затем по цепочке можно раскручивать дальше и получить, что:
`dLoss/dW = dLoss/df_k * df_k/dW = dLoss/df_k * df_k/df_k-1 * df_k-1/dW` и т.д.

При этом если считать, что каждая функция `f_k` - это некая вершина в цепочке вычислений, то получается, что в
процессе `forward` мы идем от самой первой вершины с входным значением `x` к последней вершине `f_k`, постепенно
применяя функции `f` к промежуточным результатам. Т.е. движемся слева направо.
А в процессе `backward` для вычисления градиента мы можем идти справа налево и считать накопленный градиент для каждой
вершины и передавать его предыдущим вершинам для последующего использования.

Для самой последней вершины `Loss` (которая получается после применения функции потерь) изначально нет никакого
накопленного
градиента. Мы считаем в этой вершине градиент `dLoss/df_k` и передаем его предыдущей вершине `f_k`.
В этой вершине `f_k` мы считаем градиент `df_k/df_k-1`, умножаем его на полученный градиент `dLoss/df_k` и передаем его
предыдущей вершине и так далее.

Если в какой-то момент мы дошли до вершины `f_w`, которая напрямую использует веса (которые мы хотим оптимизировать), то
на данном шаге у нас есть вся информация для того, чтобы наконец-то посчитать полный градиент `dLoss/dW`. Для этого у
нас уже посчитан весь накопленный градиент, который нам просто нужно умножить на `df_w/dW`. Т.к. `f_w` напрямую зависит
от весов `W`, то этот градиент посчитать совсем несложно.

Чтобы не пропустить все веса и посчитать градиенты для них для всех, проводится полный `backward` до самой начальной
вершины с нашим входом `X`. Таким образом весь накопленный градиент в этой вершине - это будет `dLoss/dX`.

Т.е. в итоге, чтобы посчитать градиенты по весам `W` (`dLoss/dW`), мы запускаем подсчет градиента `dLoss/dX`, в процессе
которого сможем посчитать необходимые нам градиенты.

После метода `backward`, во время которого мы подсчитаем все градиенты для наших весов, необходимо будет их всех
обновить согласно формуле градиентного спуска. Готово, итерация завершена.

(п.с. нужно не забыть занулить накопленные градиенты у весов, чтобы следующая итерация была "чистая")

Стоит также заметить, что полный градиент по какому-то конкретному параметру вектора `X` считается так:
`dLoss/dX_j = SUM(dLoss/dy_i * dy_i/dX_j for i in range(n))`.
При этом если `Y_i` зависит только от `X_i`, а от остальных нет, то `dy_i/dX_j = 0` и тогда формула выливается в:
`dLoss/dX_j = dLoss/dy_i * dy_i/dX_i`, что в общем случае для вектора `X` дает: `dLoss/dX = dLoss/dy * dy/dX`.

В целях упрощения `dLoss/dy` я буду просто заменять на `grad` (накопленный градиент).

#### Linear

Функция `forward` линейного слоя выглядит просто: `y = W * x + b`, где `x` - наш вход, `W` - матрица весов, `b` - вектор
смещений. `W` и `b` - оптимизируемые параметры.

`backward` также выглядит несложно:

* `dLoss/dW = dLoss/dy * dy/dW = grad * x`, где `x` - наш вход (его нужно
  запомнить во время прогона `forward'a`), `dLoss/dy` - накопленный градиент (`grad`), который приходит на вход
  функции `backward`  от будущей вершины.
* `dLoss/db = dLoss/dy * dy/db = grad`
* `dLoss/dX = dLoss/dy * dy/dX = grad * X`

#### ReLU

* `forward`: `relu(x) = x if x > 0 else 0`
* `backward`: `dLoss/dX = dLoss/dy * dy/dX = [grad if x > 0 else 0]` (`x` - вход, который нужно было запомнить)

#### Sigmoid

* `forward`: `s(x) = 1.0 / (1.0 + np.exp(-x))`
* `backward`: `dLoss/dX = dLoss/dy * dy/dX = grad * s(x) * (1 - s(x))` (в целях оптимизации во время форварда следует
  запомнить именно `s(x)`, чтобы заново не пересчитывать сигмоиду)

#### Tanh

* `forward`: `tanh(x) = sh(x)/ch(x)`
* `backward`: `dLoss/dX = dLoss/dy * dy/dX = grad / (ch(x) ^ 2)` - (запомнить `ch(x)`)

#### Softmax

* `forward`: `softmax(x = [x0,x2,x3,...,xn-1]) = exp(x) / sum(exp(x))`
* `backward`:

```
dLoss/dX_j = SUM(dLoss/dy_i * dy_i/dX_j for i in range(n)) =
SUM(dLoss/dy_i * (-y_i * y_j) for i in range(n) if i != j) + y_j * (1 − y_j) * dLoss/dy_j =
y_j * dLoss/dy_j - SUM(dLoss/dy_i * y_i * y_j for i in range(n)) = 
y_j * [dLoss/dy_j - SUM(dLoss/dy_i * y_i for i in range(n))] =>

dLoss/dX = y * (grad - sum(grad * y))
```

#### MSE

* `forward`: `loss(x, target) = (target - x) ** 2` (без редукции)
* `backward`: `dLoss/dX = 2 * (x - target)` ('sum' редукция)

#### Cross Entropy

* `forward`: `loss(x, target) = -sum(target * log(x))` (без редукции)
* `backward`: `dLoss/dX = -target / x` ('sum' редукция)

#### Dropout

Во время `training = True`:

* `forward`: `dropout(x) = rand_mask(p) * x / (1 - p)`, где `p` - вероятность зануления элемента,
  `rand_mask(p)` - вектор из 0 и 1, показывающий какие элементы зануляются, деление на `1 - p` необходимо, чтобы
  нормировать важность каждого элемента (во время применения дропаута важность оставшихся элементов должна повыситься,
  чтобы суммарно выглядело так, будто дропаут не применялся)
* `backward`: `dLoss/dX = dLoss/dy * dy/dX = grad * rand_mask(p) / (1 - p)`, где `rand_mask(p)` - та же самая маска,
  которая была применена во время форварда (т.е. её нужно было запомнить)

#### BatchNorm

Во время `training = True`:

* `forward`: `batchnorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * W + b`, где `mean(x)` - среднее по батчу, `var(x)` -
  стандартное отклонение по батчу, `W` - матрица оптимизируемых параметров, `b` - вектор оптимизируемых смещений

* `backward`:

1. `dLoss/db = dLoss/dy * dy/db = grad`
2. `dLoss/dW = dLoss/dy * dy/dW = grad * (x - mean(x)) / sqrt(var(x) + eps)`
3. `dLoss/dX`: доступно и пошагово
   объяснено [здесь](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

#### Conv2d и MaxPool2d:

* Очень хорошо объяснено по-русски [здесь](https://habr.com/ru/company/ods/blog/344008/)

### Полезные ссылки:

Существующие реализации нейронных сетей на numpy:

* https://github.com/Nico-Curti/NumPyNet
* https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/layers/layers.py#L969-L1215
* https://github.com/nhanwei/mnist_cnn_numpy_scratch/blob/master/layers.py

Очень полезная ссылка для понимания, как и почему работает backpropagation:

* https://sgugger.github.io/a-simple-neural-net-in-numpy.html (очень полезно)

Ссылки с различными объяснениями/реализациями сверток и CNN:

* https://habr.com/ru/company/ods/blog/344008/ (на русском)
* https://habr.com/ru/company/ods/blog/344116/ (на русском)
* https://github.com/skalinin/CNN-from-Scratch/blob/master/src/cnn/numpy_model.py
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html#scipy.signal.correlate
* Convolve vs Cross-correlation: https://stackoverflow.com/questions/20036663/understanding-numpys-convolve

Объяснения по BatchNorm:

* https://github.com/christianversloot/machine-learning-articles/blob/main/batch-normalization-with-pytorch.md
* https://medium.com/deeplearningmadeeasy/everything-you-wish-to-know-about-batchnorm-6055e07fdce2
* https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
  (очень полезно)
* https://stackoverflow.com/questions/67968913/derivative-of-batchnorm2d-in-pytorch#comment120333317_67968913