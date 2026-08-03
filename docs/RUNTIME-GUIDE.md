# Anima Runtime Guide

GPU 연구와 런타임 배포 대상의 SSOT는 `deploy.targets.toml`이다. 호스트 주소,
사용자, 포트, SSH 키는 중복 기록하지 않고 로컬 `~/.ssh/config`의 `aiden`,
`summer` 별칭으로 관리한다.

## 실행

```bash
# GPU 상태 (aiden + summer)
python3 deploy.py --gpu-status

# 로컬
python3 anima_unified.py --web
```

GPU 장기 학습은 `training.toml`의 run 정의와 supervisor 정책을 단일 기준으로
사용한다. tmux 명령을 별도로 복제하지 않는다.

```bash
# origin/main의 supervisor를 aiden에 배포하고 설정된 best.pt에서 재개
python3 deploy.py --training-run nf9_v3

# GPU API를 호출하지 않고 systemd 학습 상태 확인
python3 deploy.py --training-run nf9_v3 --training-status
```

supervisor는 설정된 시간 동안 새 step이 출력되지 않으면 전체 프로세스 그룹을
종료한다. systemd가 서비스를 다시 시작하며 학습기는 atomic `best.pt`에서
복원한다. CUDA 드라이버가 Python 프로세스만 살려 둔 채 멈추는 경우에도 다음
SSH 점검을 기다리지 않고 자동 복구한다.

## 배포

```bash
python3 deploy.py --target summer                    # origin/main 배포
python3 deploy.py --target summer --model final.pt   # 코드+모델
python3 deploy.py --target summer --status           # 상태 확인
python3 deploy.py --target summer --rollback         # 직전 릴리스 롤백
```

배포는 `origin/main`과 현재 HEAD가 일치할 때만 실행된다. 커밋된 트리를 릴리스
디렉터리로 전송하고, `data`와 체크포인트를 공유 상태로 보존한 뒤 systemd user
service를 전환한다. 포트 헬스체크가 실패하면 직전 릴리스로 자동 복귀한다.

## 실행 후 헬스 체크 (필수!)

```bash
# 1. 프로세스 확인
ssh summer 'systemctl --user status anima-lab-1.service'

# 2. 포트 확인
ssh summer 'ss -tlnp | grep 8765'

# 3. 로그 확인 (에러 없는지)
ssh summer 'journalctl --user -u anima-lab-1.service -n 30'

# 4. 배포 엔진 헬스체크
python3 deploy.py --target summer --status
```

## 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| 502 Bad Gateway | 포트 8765 안 열림 | 프로세스 확인 후 재시작 |
| Address already in use | 이전 서비스/프로세스 잔존 | `systemctl --user restart anima-lab-1` |
| Garbled output (◆▨) | byte-level 모델 UTF-8 실패 | LanguageLearner fallback 사용 |
| 영어 응답 | Claude fallback | anima_unified.py 한국어 fallback 적용 |
| Φ=0.00 | cells=2 (최소) | --max-cells 64 이상 |
| 과거 대화 잔여 | shared 상태 확인 필요 | `/home/summer/services/anima-lab-1/shared` 점검 |
| 포트 충돌 | 다른 서비스가 8765 사용 | `ss -tlnp \| grep 8765`로 소유자 확인 |
| 체크포인트 못 찾음 | 모델 미배포 | `deploy.py --target summer --model PATH` |
| SSH 세미콜론 실패 | exit 255 | `bash -c "cmd"` 래핑 |

## 서버 정보

```
aiden: GPU 연구 (현재 드라이버 상태는 --gpu-status로 확인)
summer: GPU 연구 + Anima 런타임
```
