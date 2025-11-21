"""
# Project DDong

본 문서에서는 Project DDong의 구현에 관해 논의합니다.

DDong는 영상을 크게 두 개의 영역(배경, 텍스트)로 구분하고 각각을 서로 다른 방법으로 이동시킵니다.
이를 구현하기 위해 다음과 같은 주제들이 다루어져야 합니다:

- 어떻게 노이즈를 만들 것인가?
- 어떻게 특정 영역을 *이동*시킬 것인가?
- 이동 이후 비워지는 공간을 어떻게 처리할 것인가?

> 앞으로 우리가 다룰 영상을 **Unscreenshottable Video**, 또는 **USV**라고 하겠습니다.

## 노이즈 생성

대부분의 USV는 흑/백의 노이즈만을 사용하지만, 일반화를 위해 이번에는 다른 노이즈들도 허용하는 것으로 하겠습니다. 실제로 구현된 노이즈는 아래와 같습니다.

- `BernoulliNoise`
- `UniformNoise` (in dev.)
- `GaussianNoise` (in dev.)
- `SaltAndPepperNoise` (in dev.)

> 노이즈는 모두 `Noise` 추상 클래스를 상속 받습니다.

## 영역의 이동과 빈틈 메우기

영역이 어떤 형식으로든 이동하면, 이전의 공간은 비워지게 됩니다. 그러나 이 빈틈을 정확히 계산하는 것은 번거로우며, 비효율적입니다.

그러한 이유로 단순히 두 개의 서로 다른 Noise Image를 만들고, 각각을 적당히 변형한 후 텍스트 부분만 또는 배경 부분만 가져와 합치는 방식으로 영역의 이동을 다룹니다.

또한 영역의 이동을 다루기 위해 *변환기*들은 반복적으로 이미지로 대변되는 `np.array`를 변환해주어야 합니다.

> 변환기들은 모두 `Transform` 추상 클래스를 상속받습니다.
> 각각은 모두 반복 가능하며, 차례로 *다음 프레임의 이미지*을 반환합니다.

## 비디오 생성기

비디오 생성기는 텍스트 마스크를 관리하며, 텍스트 영역과 그렇지 않은 영역을 구분하여, 배경과 텍스트의 노이즈 이미지를 합치는 역할을 합니다.
"""

from transform import LinearTransform, Direction
from noise import BernoulliNoise
from video import VideoGenerator

__all__ = (
    "LinearTransform",
    "Direction",
    "BernoulliNoise",
    "VideoGenerator",
)
