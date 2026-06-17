# Mobile Cleanliness Capture

정적 프론트엔드 앱입니다. FastAPI 서버가 `frontend` 디렉터리를 `/mobile`로 서빙합니다.

```text
http://127.0.0.1:8000/mobile
```

스마트폰 카메라는 HTTPS 또는 localhost 보안 컨텍스트에서만 동작합니다. 실제 스마트폰에서 테스트하려면 FastAPI 서버를 HTTPS로 노출하거나 HTTPS 터널을 사용하세요.

## Android 앱 패키징

Capacitor로 현재 정적 프론트엔드를 Android 앱으로 패키징합니다. 앱 기본 백엔드 주소는 `app-config.js`의 `nativeBackendBase`에서 변경할 수 있습니다.

```powershell
cd frontend
npm install
npm run build
npx cap add android
npm run sync:android
```

Android Studio에서 `frontend/android`를 열어 APK를 빌드하거나, Android SDK가 설치된 환경에서 다음 명령을 실행합니다.

```powershell
npm run build:android
```

디버그 APK 경로:

```text
frontend/android/app/build/outputs/apk/debug/app-debug.apk
```

APK 빌드에는 JDK와 Android SDK가 필요합니다. `java -version`이 동작하지 않으면 Android Studio 또는 JDK 17 이상을 먼저 설치하세요.
