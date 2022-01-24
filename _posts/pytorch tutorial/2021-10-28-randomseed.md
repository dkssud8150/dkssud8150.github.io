---
layout:   post
title:    "random seed 설정"
subtitle: "randomseed를 설정하는 이유"
category: Classlog
tags:     pytorch-tutorial randomseed
---

1. this ordered seed list will be replaced by the toc
{:toc}

# Random Seed

seed() 괄호 안에 들어가는 숫자는 무슨 의미일까?

<br>

seed value 숫자 자체는 중요하지 않고 서로 다른 시드를 사용하면 서로 다른 난수를 생성한다는 점만 알면 된다.

random seed를 고정한다는 말은 동일한 셋트의 난수를 생성할 수 있게 하는 것이다.

```python
np.random.seed(0)
np.random.rand(5)
```

<br>

## pytorch random seed

딥러닝에서 초기화할때 random num을 사용한다. 이 때, 실험할 때마다 다른 값이 되면, 다른 결과값이 나타날 수 있다. 

그렇기 때문에 실험을 동일하게 진행하기 위해서는 동일한 난수의 사용이 필요하다.

pytorch에서는 random seed를 고정하기 위해 manual_seed 함수를 사용한다.

benchmark는 input size가 변하지 않을 때 사용하면 더 빠른 런타임을 사용할 수 있다. 하지만 그렇지 않을 경우 False로 설정하는 것이 좋다.

```python
import torch

torch.manual_seed(1)
torch.rand(5)
```

```python
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
```

# Reference
- 