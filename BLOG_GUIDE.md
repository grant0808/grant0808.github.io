# GitHub 블로그 사용 안내

이 저장소는 Jekyll과 Chirpy 테마로 구성된 GitHub Pages 블로그입니다.

## 1. 처음 한 번만 설치

macOS에서 Homebrew를 사용하는 경우:

```bash
brew install ruby@3.3
export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
bundle config set --local path vendor/bundle
bundle install
npm install
```

Apple Silicon이 아닌 Mac에서 Homebrew Ruby의 위치가 다르면 아래 명령으로
경로를 확인할 수 있습니다.

```bash
brew --prefix ruby@3.3
```

## 2. 로컬에서 블로그 실행

터미널을 새로 열었다면 먼저 Ruby 경로를 적용합니다.

```bash
export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
```

그다음 테마의 CSS·JavaScript를 만들고 Jekyll 서버를 실행합니다.

```bash
npm run build
bundle exec jekyll serve --livereload
```

브라우저에서 <http://127.0.0.1:4000>을 열면 블로그를 볼 수 있습니다.
서버는 `Ctrl+C`로 종료합니다.

## 3. 새 글 작성

파일 이름은 `_posts/YYYY-MM-DD-영문-슬러그.md` 형식을 사용합니다.

예:

```text
_posts/2026-07-28-my-first-post.md
```

글의 기본 형식:

```markdown
---
layout: post
title: "첫 번째 글"
date: 2026-07-28 10:00 +0900
description: 글 목록과 검색 결과에 표시할 설명
categories: [Development]
tags: [GitHub, Jekyll]
pin: false
math: false
mermaid: false
---

여기부터 Markdown으로 본문을 작성합니다.
```

`jekyll-compose`로 파일을 자동 생성할 수도 있습니다.

```bash
bundle exec jekyll post "my-first-post"
```

생성된 파일의 날짜, 제목, 카테고리와 태그를 확인한 뒤 작성합니다.

이미지는 `assets/img` 아래에 저장하고 다음처럼 넣습니다.

```markdown
![이미지 설명](/assets/img/example.png)
```

## 4. 블로그 정보 변경

`_config.yml`에서 다음 항목을 수정합니다.

- `title`, `tagline`, `description`: 블로그 이름과 설명
- `url`, `github.username`: 블로그 주소와 GitHub 계정
- `social`: 작성자 이름, 이메일, 소셜 링크
- `avatar`: 프로필 이미지 경로
- `comments`: Giscus 댓글 설정

설정 파일을 바꾼 뒤에는 실행 중인 Jekyll 서버를 껐다가 다시 시작해야
변경 내용이 정확히 반영됩니다.

## 5. GitHub Pages에 배포

이 저장소의 `.github/workflows/jekyll.yml`이 `master` 브랜치에 변경 사항이
올라오면 자동으로 사이트를 빌드하고 배포합니다.

```bash
git add .
git commit -m "docs: add a new blog post"
git push origin master
```

처음 한 번은 GitHub 저장소의 **Settings → Pages → Build and deployment →
Source**에서 **GitHub Actions**를 선택합니다. 배포 진행 상황은 저장소의
**Actions** 탭에서 확인할 수 있습니다.

배포가 끝나면 <https://grant0808.github.io>에서 확인합니다. 반영에는 잠시
시간이 걸릴 수 있습니다.

## 6. 자주 겪는 문제

- `Your Ruby version is ...` 오류: Ruby 3.3 경로가 `PATH` 앞쪽에 있는지 확인합니다.
- `Could not find gem` 오류: `bundle install`을 다시 실행합니다.
- CSS 또는 JavaScript가 깨짐: `npm install` 후 `npm run build`를 실행합니다.
- 글이 보이지 않음: 파일 이름의 날짜, 머리말의 `date`, 현재 시간과 시간대를
  확인합니다. 미래 날짜의 글은 기본적으로 표시되지 않습니다.
- 배포 실패: GitHub 저장소의 **Actions** 탭에서 실패한 단계의 로그를 확인합니다.
