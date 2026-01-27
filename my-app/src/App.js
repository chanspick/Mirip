import { useEffect } from "react";
import {
  Routes,
  Route,
  useNavigationType,
  useLocation,
} from "react-router-dom";
// MIRIP 프로토타입 페이지
import Landing from "./pages/Landing/Landing";
// SPEC-COMP-001: 공모전 페이지
import CompetitionList from "./pages/competitions/CompetitionList";
import CompetitionDetail from "./pages/competitions/CompetitionDetail";
import SubmitPage from "./pages/competitions/SubmitPage";
// AI 진단 페이지
import DiagnosisPage from "./pages/diagnosis/DiagnosisPage";
// SPEC-CRED-001: 마이페이지 및 공개 프로필
import { ProfilePage } from "./pages/Profile";
import PublicProfilePage from "./pages/PublicProfile";
// SPEC-CRED-001: M4 포트폴리오 관리
import PortfolioPage from "./pages/Portfolio";
// 기존 페이지 (추후 정리 예정)
import MiripHome from "./pages/MiripHome";
import MiripComp from "./pages/MiripComp";
import MiripEdu from "./pages/MiripEdu";

function App() {
  const action = useNavigationType();
  const location = useLocation();
  const pathname = location.pathname;

  useEffect(() => {
    if (action !== "POP") {
      window.scrollTo(0, 0);
    }
  }, [action, pathname]);

  useEffect(() => {
    let title = "";
    let metaDescription = "";

    switch (pathname) {
      case "/":
        title = "MIRIP - AI 기반 예술/디자인 진로 진단 플랫폼";
        metaDescription = "AI가 분석하는 나만의 예술/디자인 진로. 프로토타입을 체험해보세요.";
        break;
      case "/competitions":
        title = "공모전 - MIRIP";
        metaDescription = "다양한 분야의 공모전에 참여하고 실력을 뽐내보세요.";
        break;
      case "/diagnosis":
        title = "AI 진단 - MIRIP";
        metaDescription = "작품 이미지를 업로드하고 AI가 분석하는 대학별 합격 가능성을 확인하세요.";
        break;
      case "/profile":
        title = "마이페이지 - MIRIP";
        metaDescription = "나의 활동 현황과 잔디밭을 확인하세요.";
        break;
      case "/portfolio":
        title = "포트폴리오 - MIRIP";
        metaDescription = "나의 작품들을 관리하고 공유하세요.";
        break;
      case "/legacy/home":
        title = "MIRIP Home";
        metaDescription = "";
        break;
      case "/mirip-comp":
        title = "";
        metaDescription = "";
        break;
      case "/mirip-edu":
        title = "";
        metaDescription = "";
        break;
      default:
        // 동적 라우트 처리 (공모전 상세, 출품 페이지, 공개 프로필)
        if (pathname.startsWith("/competitions/")) {
          title = "공모전 - MIRIP";
          metaDescription = "공모전 상세 정보를 확인하세요.";
        } else if (pathname.startsWith("/profile/")) {
          title = "프로필 - MIRIP";
          metaDescription = "사용자의 공개 프로필을 확인하세요.";
        }
        break;
    }

    if (title) {
      document.title = title;
    }

    if (metaDescription) {
      const metaDescriptionTag = document.querySelector(
        'head > meta[name="description"]'
      );
      if (metaDescriptionTag) {
        metaDescriptionTag.content = metaDescription;
      }
    }
  }, [pathname]);

  return (
    <Routes>
      {/* MIRIP 프로토타입 Landing 페이지 */}
      <Route path="/" element={<Landing />} />
      {/* SPEC-COMP-001: 공모전 페이지 */}
      <Route path="/competitions" element={<CompetitionList />} />
      <Route path="/competitions/:id" element={<CompetitionDetail />} />
      <Route path="/competitions/:id/submit" element={<SubmitPage />} />
      {/* AI 진단 페이지 */}
      <Route path="/diagnosis" element={<DiagnosisPage />} />
      {/* SPEC-CRED-001: 마이페이지 및 공개 프로필 */}
      <Route path="/profile" element={<ProfilePage />} />
      <Route path="/profile/:username" element={<PublicProfilePage />} />
      {/* SPEC-CRED-001: M4 포트폴리오 관리 */}
      <Route path="/portfolio" element={<PortfolioPage />} />
      {/* 기존 페이지 (legacy 경로로 이동) */}
      <Route path="/legacy/home" element={<MiripHome />} />
      <Route path="/mirip-comp" element={<MiripComp />} />
      <Route path="/mirip-edu" element={<MiripEdu />} />
    </Routes>
  );
}
export default App;
