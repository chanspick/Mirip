import { useEffect } from "react";
import {
  Routes,
  Route,
  useNavigationType,
  useLocation,
} from "react-router-dom";
// SPEC-FIREBASE-001: Landing 페이지 (사전등록)
import Landing from "./pages/Landing/Landing";
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
        metaDescription = "AI가 분석하는 나만의 예술/디자인 진로. 사전등록하고 런칭 소식을 받아보세요.";
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
      {/* SPEC-FIREBASE-001: 사전등록 Landing 페이지 */}
      <Route path="/" element={<Landing />} />
      {/* 기존 페이지 (legacy 경로로 이동) */}
      <Route path="/legacy/home" element={<MiripHome />} />
      <Route path="/mirip-comp" element={<MiripComp />} />
      <Route path="/mirip-edu" element={<MiripEdu />} />
    </Routes>
  );
}
export default App;
