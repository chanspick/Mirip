/**
 * 공모전 시드 데이터 스크립트
 *
 * 테스트용 공모전 5개와 출품작 20개를 생성합니다.
 * 브라우저 콘솔에서 실행하거나, 개발 환경에서 import하여 사용합니다.
 *
 * 사용법: seedCompetitions()
 *
 * @module scripts/seedCompetitions
 */

import { collection, addDoc, Timestamp } from 'firebase/firestore';
import { db } from '../config/firebase';

/**
 * 테스트 공모전 데이터
 */
const competitionsData = [
  {
    title: '2025 대한민국 시각디자인 공모전',
    description: `대한민국을 대표하는 시각디자인 공모전입니다.

창의적인 아이디어와 뛰어난 시각적 표현력을 가진 작품을 모집합니다.
브랜딩, 포스터, 패키지 등 다양한 분야의 시각디자인 작품을 출품해주세요.

[참가 자격]
- 전국 예술/디자인 전공 대학(원)생
- 시각디자인에 관심 있는 일반인

[심사 기준]
- 창의성 30%
- 완성도 30%
- 메시지 전달력 20%
- 기술력 20%`,
    thumbnail: 'https://images.unsplash.com/photo-1561070791-2526d30994b5?w=800',
    category: 'visual_design',
    organizer: '한국디자인진흥원',
    prize: 10000000,
    startDate: Timestamp.fromDate(new Date('2025-01-01')),
    endDate: Timestamp.fromDate(new Date('2025-02-28')),
    rules: '1인 1작품만 출품 가능\n표절작 발견 시 수상 취소\n출품작의 저작권은 참가자에게 있음',
    schedule: [
      { date: '2025-01-01', title: '접수 시작', description: '온라인 접수 오픈' },
      { date: '2025-02-15', title: '접수 마감', description: '오후 6시까지' },
      { date: '2025-02-28', title: '결과 발표', description: '홈페이지 공지' },
    ],
    participantCount: 156,
  },
  {
    title: '제5회 산업디자인 아이디어 챌린지',
    description: `혁신적인 제품 아이디어를 찾습니다!

일상의 불편함을 해결하는 창의적인 산업디자인 아이디어를 모집합니다.
스케치, 3D 렌더링, 프로토타입 등 다양한 형태로 출품 가능합니다.

[주제]
지속가능한 미래를 위한 제품 디자인

[참가 자격]
- 제한 없음 (개인 또는 3인 이하 팀)`,
    thumbnail: 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800',
    category: 'industrial_design',
    organizer: '산업통상자원부',
    prize: 15000000,
    startDate: Timestamp.fromDate(new Date('2025-01-15')),
    endDate: Timestamp.fromDate(new Date('2025-03-31')),
    rules: '팀 참가 시 대표자 1인 등록\n기출품작 불가\n수상작 전시 동의 필수',
    schedule: [
      { date: '2025-01-15', title: '공모 시작' },
      { date: '2025-03-15', title: '1차 서류 심사' },
      { date: '2025-03-31', title: '최종 발표' },
    ],
    participantCount: 89,
  },
  {
    title: '전통공예 현대화 공모전',
    description: `전통과 현대의 조화를 담은 공예 작품을 모집합니다.

한국 전통 공예 기법을 현대적으로 재해석한 작품을 출품해주세요.
도자기, 목공예, 금속공예, 섬유공예 등 모든 분야 참가 가능합니다.

[심사 기준]
- 전통성 계승 25%
- 현대적 해석 25%
- 실용성 25%
- 완성도 25%`,
    thumbnail: 'https://images.unsplash.com/photo-1565193566173-7a0ee3dbe261?w=800',
    category: 'craft',
    organizer: '문화체육관광부',
    prize: 8000000,
    startDate: Timestamp.fromDate(new Date('2024-12-01')),
    endDate: Timestamp.fromDate(new Date('2025-01-25')),
    rules: '직접 제작한 작품만 출품 가능\n작품 사진 5장 이상 제출',
    schedule: [
      { date: '2024-12-01', title: '접수 시작' },
      { date: '2025-01-20', title: '접수 마감' },
      { date: '2025-01-25', title: '결과 발표' },
    ],
    participantCount: 234,
  },
  {
    title: '신진작가 회화 공모전',
    description: `미래의 거장을 발굴합니다!

신진 작가들의 창작 활동을 지원하기 위한 회화 공모전입니다.
유화, 수채화, 아크릴화, 한국화 등 모든 회화 장르 참가 가능합니다.

[참가 자격]
- 만 35세 이하
- 미술 전공자 또는 1년 이상 작품 활동 경력자

[특전]
- 대상 수상자 개인전 지원`,
    thumbnail: 'https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=800',
    category: 'fine_art',
    organizer: '국립현대미술관',
    prize: 20000000,
    startDate: Timestamp.fromDate(new Date('2025-02-01')),
    endDate: Timestamp.fromDate(new Date('2025-04-30')),
    rules: '100호 이내 작품\n액자 포함 출품\n설치비 본인 부담',
    schedule: [
      { date: '2025-02-01', title: '접수 시작' },
      { date: '2025-04-15', title: '접수 마감' },
      { date: '2025-04-30', title: '시상식' },
    ],
    participantCount: 0,
  },
  {
    title: '친환경 패키지 디자인 공모전',
    description: `지구를 생각하는 패키지 디자인을 찾습니다.

플라스틱 사용을 줄이고 재활용이 가능한 친환경 패키지 디자인을 모집합니다.
식품, 화장품, 생활용품 등 다양한 카테고리에서 출품 가능합니다.

[심사 기준]
- 환경친화성 40%
- 디자인 완성도 30%
- 실용성 20%
- 혁신성 10%`,
    thumbnail: 'https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?w=800',
    category: 'visual_design',
    organizer: '환경부',
    prize: 5000000,
    startDate: Timestamp.fromDate(new Date('2024-11-01')),
    endDate: Timestamp.fromDate(new Date('2024-12-31')),
    rules: '실제 제작 가능한 디자인\n재료 명세 포함 필수',
    schedule: [
      { date: '2024-11-01', title: '접수 시작' },
      { date: '2024-12-15', title: '접수 마감' },
      { date: '2024-12-31', title: '결과 발표' },
    ],
    participantCount: 312,
  },
];

/**
 * 테스트 출품작 생성 함수
 * @param {string} competitionId - 공모전 ID
 * @param {number} count - 생성할 출품작 수
 */
const generateSubmissions = (competitionId, count) => {
  const submissions = [];
  const titles = [
    '도시의 빛',
    '자연의 숨결',
    '미래로 가는 길',
    '기억의 조각',
    '새로운 시작',
    '조화와 균형',
    '변화의 순간',
    '내면의 풍경',
    '시간의 흐름',
    '꿈꾸는 공간',
  ];

  for (let i = 0; i < count; i++) {
    submissions.push({
      competitionId,
      userId: `test_user_${i + 1}`,
      title: titles[i % titles.length],
      description: `이 작품은 ${titles[i % titles.length]}을(를) 주제로 제작되었습니다.`,
      imageUrl: `https://picsum.photos/800/600?random=${competitionId}_${i}`,
      tools: ['포토샵', 'Illustrator', '연필', '수채화'][i % 4],
      duration: `${(i % 4) + 1}주`,
      status: i === 0 ? 'winner' : 'approved',
      createdAt: Timestamp.now(),
      updatedAt: Timestamp.now(),
    });
  }

  return submissions;
};

/**
 * 시드 데이터 생성 실행
 */
export const seedCompetitions = async () => {
  console.log('🌱 시드 데이터 생성 시작...');

  try {
    const competitionIds = [];

    // 공모전 생성
    for (const competition of competitionsData) {
      const docRef = await addDoc(collection(db, 'competitions'), {
        ...competition,
        createdAt: Timestamp.now(),
        updatedAt: Timestamp.now(),
      });
      competitionIds.push(docRef.id);
      console.log(`✅ 공모전 생성: ${competition.title} (${docRef.id})`);
    }

    // 출품작 생성 (공모전당 4개씩 = 총 20개)
    let submissionCount = 0;
    for (const competitionId of competitionIds) {
      const submissions = generateSubmissions(competitionId, 4);
      for (const submission of submissions) {
        await addDoc(collection(db, 'submissions'), submission);
        submissionCount++;
      }
    }
    console.log(`✅ 출품작 ${submissionCount}개 생성 완료`);

    console.log('🎉 시드 데이터 생성 완료!');
    console.log(`- 공모전: ${competitionIds.length}개`);
    console.log(`- 출품작: ${submissionCount}개`);

    return { competitionIds, submissionCount };
  } catch (error) {
    console.error('❌ 시드 데이터 생성 실패:', error);
    throw error;
  }
};

export default seedCompetitions;
