# AI_Natural-language-processing
1. 데이터 수집 및 전처리 , 정규표현식, 신경망
2. Word2Vec, Embedding, RNN, LSTM, Seq2Seq, Attention
3. NLTK, KoNLPy, 텍스트분류, 감성분석, 문서요약, 질의응답모델
4. 자연어 처리 프로젝트 (개인)


# 환경 설정하기
## Anaconda 에 텐서플로 버전 2.3 설치하기 
Anaconda Prompt를 관리자 권한으로 실행하고 아래 명령을 차례로 실행한다
python -m pip install --upgrade pip
pip3 install tensorflow==2.3.0

## [설치 버전 확인] 쥬피터 노트북 셀에서 에서 아래 코드를 실행시켜 버전을 확인해본다
* 텐서플로 버전을 알아본다
import tensorflow as tf
tf.__version__
'2.3.0' 이 출력되면 설치 성공


## Anaconda 텐서플로 설치 가상 환경 만들기
Anaconda Prompt를 관리자 권한으로 실행하고 아래 명령을 실행한다
conda create -n tf230 python=3.8 anaconda
conda activate tf230
python -m pip install --upgrade pip
pip3 install tensorflow==2.3.0

## [설치 버전 확인] 쥬피터 노트북 셀에서 에서 아래 코드를 실행시켜 버전을 확인해본다
* 텐서플로 버전을 알아본다
import tensorflow as tf
tf.__version__
(주의: 언더바가 두개임)
'2.3.0' 이 출력되면 설치 성공!!

----------------------------------------------------------------------------------------------
# 윈도우에 PyCharm 설치 환경에 텐서플로 설치하기

1] python 3.7.6 설치
https://www.python.org/downloads/release/python-376/

사이트에서 하단 Files 목록에서 Windows x86-64 executable installer을 선택 클릭하여
'python-3.7.6-amd64.exe' 파일을 다운로드 받아 실행 시킨다
설치 시작 화면에서 맨 아래 'Add Python 3.7 to PATH' 를 체크해주고 'Install Now'를 선택하고
다른 옵션 선택없이 그대로 설치를 진행한다
설치가 끝나면 윈도우 시작메뉴에 Pyhon 3.7 이 새로 설치되어 추가된걸 확인 할 수 있을 것이다

[2] PyCharm 다운로드 및 설치
https://www.jetbrains.com/ko-kr/pycharm/download/#section=windows

에서 오른쪽의 'Community' 의 '다운로드' 버튼을 눌러 'pycharm-community-2020.1.exe' 파일을 다운받아 실행시킨다
일부 옵션(Path)을 선택하면서 설치해준다

[3] tensorflow 설치하기
아래 주소를 참조하여 설치한다 tensorflow를 설치 하도록 한다
https://webnautes.tistory.com/1173

PyCharm에서 세 프로젝트를 만들고 하단의 'Terminal'을 눌러 아래와 같이 설치하면 된다
python -m pip install --upgrade pip
pip3 install tensorflow==2.0.0

-------------------------------------------------------------------------------------------------

* 쥬피터 노트북에 tensorflow 설치 방법
python =3.8.x
tensoflow = 2.3.0

[1] Anaconda 파이썬 버전 변경하기(파이썬 버전이 3.7.x 아래이거나 버전이 낮아 설치 안되는경우만 실행)
(파이썬 버전이 3.8.x 이상인 분은 [2] 번 부터 시작하세요)

* Anaconda Prompt를 관리자 권한으로 실행하고
conda create -n py383 python=3.8.3 anaconda # 몇 분 소요됨
conda activate py383

* 파이썬 버전 확인 명령
python -V

* 만일 원래 파이썬 버전으로 다시 사용하고 싶을 때는 deactivate를 실행한다
conda deactivate py383

[2] Anaconda Prompt를 관리자 권한으로 실행하고 아래 명령을 실행한다
텐서플로 설치
python -m pip install --upgrade pip
pip3 install tensorflow==2.3.0

[3] 쥬피터 노트북 셀에서 에서 아래 코드를 실행시켜 버전을 확인해본다
* 파이썬 버전을 알아본다
import sys
print(sys.version)
* 텐서플로 버전을 알아본다
import tensorflow as tf
tf.__version__

'2.3.0' 이 출력되면 설치 성공!!

* AVX 미지원으로 인한 "DLL 초기화 루틴을 실행할 수 없습니다." 에러 해결방법
https://datamod.tistory.com/139

* 사용하고 있는 CPU 기술 탭의 Intel(R) Advanced Vector Extensions지원 여부는
pidkor47.msi(강사 깃허브에서)
혹은 https://downloadcenter.intel.com/ko/download/28539?v=t 에서 다운로드 설치하여 확인힐 수 있다

* 파이썬 버전을 3.6.8로 추가 설치한다
* Anaconda Prompt를 관리자 권한으로 실행하고
conda create -n py368 python=3.6.8 anaconda # 몇 분 소요됨
conda activate py368

* 파이썬 버전 확인 명령
python -V

* 텐서플로를 AVX를 지원하지 않는 1.6 보다 낮은 버전(1.5.0)으로 설치 한다
python -m pip install --upgrade pip
pip3 install tensorflow==1.5.0 # 장시간 소요됨

* 아나콘다 가상환경 제거
conda env list
conda remove --name py383 --all
----------------------------------------------------------------------------------------------------------


Anaconda 텐서플로 버전 2.3.0 설치 가상 환경 만들기
Anaconda Prompt를 관리자 권한으로 실행하고 아래 명령을 실행한다
conda create -n tf230 python=3.7 anaconda
conda activate tf20
python -m pip install --upgrade pip
pip3 install tensorflow==2.3.0

설치가 완료되면 윈도우 시작메뉴의 'Anaconda3 (64-bit)' 안의 'Jupyter Notebook (tf230)'을 실행시킨다
아래 코드를 실행 시켜 버전을 확인해 본다
import sys
sys.version
import tensorflow as tf
tf.version
'2.3.0' 이 출력되면 설치 성공!!

(tensorflow 버전 1.x 용 소스코드는 버전 2.x와 호환되지 않는게 API가 많아서 실행 오류가 날 것이다)
https://www.tensorflow.org/overview/?hl=ko
의 2.x 예제 소스를 복사하여 붙여 넣어 실행 시켜본다
-------------------------------------------------------------------------------------------------

- GitHub의 쥬피터 노트북 파일 다운로드 방법
https://datascience.stackexchange.com/questions/35555/how-to-download-a-jupyter-notebook-from-github

git에서 노트북 파일을 클릭하고 --> Raw 버튼 클릭 --> Ctrl + S 로 저장 --> 저장시 확장자 .txt를 없애준다 --> 주피터노트북에서 읽어온다

쥬피터 노트북 단축키 요약
https://kkokkilkon.tistory.com/151


쥬피터 노트북 폰트 변경
https://bryan7.tistory.com/1060
'custom.css' 파일 사용
