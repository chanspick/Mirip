import { useCallback } from "react";
import BackgroundHorizontalBorder from "../components/BackgroundHorizontalBorder";
import { useNavigate } from "react-router-dom";
import styles from "./MiripHome.module.css";

const MiripHome = () => {
  const navigate = useNavigate();

  const onLinkContainerClick = useCallback(() => {
    navigate("/mirip-comp");
  }, [navigate]);

  const onButtonContainerClick = useCallback(() => {
    navigate("/mirip-edu");
  }, [navigate]);

  return (
    <div className={styles.miripHome}>
      <main className={styles.container}>
        <BackgroundHorizontalBorder
          mainLogob9ffbb6svg="/mainlogob9ffbb6svg1.svg"
          linkWidth="128px"
          aColor="#5d6970"
          containerColor="#5d6970"
          buttonWidth="385px"
          linkBoxShadow="0px 4px 4px rgba(0, 0, 0, 0.25)"
          linkBackgroundColor="#fff"
          linkBorder="1px solid #000"
        />
        <section className={styles.main}>
          <div className={styles.container1}>
            <div className={styles.container2}>
              <div className={styles.ongoing}>
                <div className={styles.container3}>
                  <div className={styles.container4}>
                    <div className={styles.heading3}>
                      <div className={styles.link}>
                        <a className={styles.a}>진행중인 대회 🏆</a>
                        <div className={styles.margin}>
                          <div className={styles.container5}>
                            <img
                              className={styles.icon}
                              loading="lazy"
                              alt=""
                              src="/icon1.svg"
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.container6}>
                      <div className={styles.div}>
                        미립 주최 혹은 소개 공모전들입니다
                      </div>
                    </div>
                  </div>
                  <div className={styles.container7}>
                    <div className={styles.button}>
                      <b className={styles.viewAll}>1</b>
                    </div>
                    <div className={styles.button1}>
                      <b className={styles.viewAll}>2</b>
                    </div>
                    <div className={styles.button2}>
                      <b className={styles.b1}>전체 보기</b>
                    </div>
                  </div>
                </div>
                <div className={styles.contestList}>
                  <div className={styles.link1} onClick={onLinkContainerClick}>
                    <div className={styles.background}>
                      <img
                        className={styles.cardCptjpegIcon}
                        alt=""
                        src="/card-cptjpeg@2x.png"
                      />
                      <div className={styles.container8}>
                        <div className={styles.container9}>
                          <div className={styles.container10}>
                            <div className={styles.background1}>
                              <img
                                className={styles.picturejpegIcon}
                                loading="lazy"
                                alt=""
                                src="/picturejpeg@2x.png"
                              />
                            </div>
                            <div className={styles.margin1}>
                              <div className={styles.background2}>
                                <div className={styles.container11}>
                                  <div className={styles.jo}>jo</div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin2}>
                              <div className={styles.background3}>
                                <div className={styles.strong}>
                                  <div className={styles.div1}>딴영</div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.marginIcon}>
                              <div className={styles.background3}>
                                <div className={styles.strong}>
                                  <div className={styles.div1}>산산</div>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div className={styles.margin4}>
                            <div className={styles.overlay}>
                              <div className={styles.div3}>219명 참여중</div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.background5}>
                        <img
                          className={styles.logo1jpegIcon}
                          loading="lazy"
                          alt=""
                          src="/logo1jpeg@2x.png"
                        />
                        <div className={styles.separatorWrapper}>
                          <div className={styles.separator} />
                        </div>
                        <a
                          className={styles.a1}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          상금 6,300만원
                        </a>
                      </div>
                    </div>
                    <div className={styles.contestDescription}>
                      <a
                        className={styles.nh}
                        href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                        target="_blank"
                      >
                        2024 NH 투자증권 빅데이터 경진대회
                      </a>
                      <div className={styles.progress} />
                      <div className={styles.container14}>
                        <a
                          className={styles.a2}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          2024.09.02
                        </a>
                        <div className={styles.statusSeparator} />
                        <a
                          className={styles.openD12}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          OPEN D-12
                        </a>
                      </div>
                    </div>
                    <div className={styles.container15}>
                      <div className={styles.overlay1}>
                        <a
                          className={styles.a3}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          경진대회
                        </a>
                      </div>
                      <div className={styles.background6}>
                        <a
                          className={styles.a3}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          빅데이터
                        </a>
                      </div>
                      <div className={styles.background6}>
                        <a
                          className={styles.nh1}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          NH
                        </a>
                      </div>
                      <div className={styles.background6}>
                        <a
                          className={styles.a5}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          분석
                        </a>
                      </div>
                      <div className={styles.background6}>
                        <a
                          className={styles.ai}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          AI
                        </a>
                      </div>
                      <div className={styles.background6}>
                        <a
                          className={styles.etf}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          ETF
                        </a>
                      </div>
                      <div className={styles.background11}>
                        <a
                          className={styles.microsoft}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          Microsoft
                        </a>
                      </div>
                      <div className={styles.background12}>
                        <a
                          className={styles.microsoft}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          Tableau
                        </a>
                      </div>
                      <div className={styles.background13}>
                        <a
                          className={styles.microsoft}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          시각화
                        </a>
                      </div>
                      <div className={styles.background14}>
                        <a
                          className={styles.microsoft}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          아이디어
                        </a>
                      </div>
                    </div>
                  </div>
                  <div className={styles.cardList}>
                    <div className={styles.threeCards}>
                      <div className={styles.background}>
                        <img
                          className={styles.cardCptjpegIcon}
                          alt=""
                          src="/card-cptjpeg-1@2x.png"
                        />
                        <div className={styles.container8}>
                          <div className={styles.container17}>
                            <div className={styles.container10}>
                              <div className={styles.background1}>
                                <img
                                  className={styles.picturejpegIcon}
                                  alt=""
                                  src="/picturejpeg-1@2x.png"
                                />
                              </div>
                              <div className={styles.margin5}>
                                <div className={styles.background17}>
                                  <div className={styles.strong}>
                                    <div className={styles.ll}>ll</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.margin6}>
                                <div className={styles.background18}>
                                  <div className={styles.container11}>
                                    <div className={styles.an}>an</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.marginIcon}>
                                <div className={styles.background19}>
                                  <img
                                    className={styles.picturejpegIcon}
                                    alt=""
                                    src="/picturejpeg-2@2x.png"
                                  />
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin4}>
                              <div className={styles.overlay}>
                                <div className={styles.div4}>
                                  1,112명 참여중
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className={styles.background20}>
                          <div className={styles.logo1jpegWrapper}>
                            <img
                              className={styles.logo1jpegIcon1}
                              alt=""
                              src="/logo1jpeg-1@2x.png"
                            />
                          </div>
                          <div className={styles.rectangleWrapper}>
                            <div className={styles.separator} />
                          </div>
                          <div className={styles.div5}>상금 2,200만원</div>
                        </div>
                      </div>
                      <div className={styles.contestDescription}>
                        <b className={styles.ai1}>제2회 신약개발 AI 경진대회</b>
                        <div className={styles.progress1}>
                          <div className={styles.container21}>
                            <div className={styles.background21} />
                          </div>
                        </div>
                        <div className={styles.container14}>
                          <div className={styles.empty}>2024.08.05</div>
                          <div className={styles.statusSeparator} />
                          <div className={styles.d33}>D-33</div>
                        </div>
                      </div>
                      <div className={styles.container15}>
                        <div className={styles.overlay1}>
                          <div className={styles.div6}>경진대회</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.div6}>알고리즘</div>
                        </div>
                        <div className={styles.background23}>
                          <div className={styles.div8}>분자 구조</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.nlp}>정형</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.nlp}>회귀</div>
                        </div>
                        <div className={styles.background13}>
                          <div className={styles.nrmse}>바이오</div>
                        </div>
                        <div className={styles.background27}>
                          <div className={styles.nrmse}>NRMSE</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.threeCards}>
                      <div className={styles.background}>
                        <img
                          className={styles.cardCptjpegIcon}
                          alt=""
                          src="/card-cptjpeg-2@2x.png"
                        />
                        <div className={styles.container8}>
                          <div className={styles.container9}>
                            <div className={styles.container10}>
                              <div className={styles.background29}>
                                <div className={styles.container11}>
                                  <div className={styles.jo}>pi</div>
                                </div>
                              </div>
                              <div className={styles.margin5}>
                                <div className={styles.background19}>
                                  <img
                                    className={styles.picturejpegIcon}
                                    alt=""
                                    src="/picturejpeg-3@2x.png"
                                  />
                                </div>
                              </div>
                              <div className={styles.margin6}>
                                <div className={styles.background31}>
                                  <div className={styles.container11}>
                                    <div className={styles.jo}>hj</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.marginIcon}>
                                <div className={styles.background19}>
                                  <img
                                    className={styles.picturejpegIcon}
                                    alt=""
                                    src="/picturejpeg-4@2x.png"
                                  />
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin4}>
                              <div className={styles.overlay}>
                                <div className={styles.div3}>382명 참여중</div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className={styles.background33}>
                          <img
                            className={styles.logo1jpegIcon2}
                            alt=""
                            src="/logo1jpeg-2@2x.png"
                          />
                          <div className={styles.rectangleContainer}>
                            <div className={styles.separator} />
                          </div>
                          <div className={styles.wrapper}>
                            <div className={styles.div5}>상금 2,100만원</div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.contestDescription}>
                        <b className={styles.samsungAiChallenge}>
                          2024 Samsung AI Challenge : Mac…
                        </b>
                        <div className={styles.progress1}>
                          <div className={styles.container21}>
                            <div className={styles.background34} />
                          </div>
                        </div>
                        <div className={styles.container14}>
                          <div className={styles.empty}>2024.08.01</div>
                          <div className={styles.statusSeparator} />
                          <div className={styles.d33}>D-23</div>
                        </div>
                      </div>
                      <div className={styles.container31}>
                        <div className={styles.overlay1}>
                          <div className={styles.div6}>경진대회</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.div6}>알고리즘</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.nlp}>채용</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.div18}>반도체</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.nlp}>정형</div>
                        </div>
                        <div className={styles.background39}>
                          <div className={styles.nrmse}>회귀</div>
                        </div>
                        <div className={styles.background40}>
                          <div className={styles.nrmse}>EF</div>
                        </div>
                        <div className={styles.background41}>
                          <div className={styles.oodAuroc}>OOD AUROC</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.threeCards}>
                      <div className={styles.background}>
                        <img
                          className={styles.cardCptjpegIcon}
                          alt=""
                          src="/card-cptjpeg-3@2x.png"
                        />
                        <div className={styles.container8}>
                          <div className={styles.container9}>
                            <div className={styles.container10}>
                              <div className={styles.background43}>
                                <div className={styles.strong}>
                                  <div className={styles.div1}>진짜</div>
                                </div>
                              </div>
                              <div className={styles.margin5}>
                                <div className={styles.background44}>
                                  <div className={styles.strong}>
                                    <div className={styles.co}>00</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.margin2}>
                                <div className={styles.background19}>
                                  <img
                                    className={styles.picturejpegIcon}
                                    alt=""
                                    src="/picturejpeg-5@2x.png"
                                  />
                                </div>
                              </div>
                              <div className={styles.margin15}>
                                <div className={styles.background46}>
                                  <div className={styles.container11}>
                                    <div className={styles.my}>my</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin4}>
                              <div className={styles.overlay}>
                                <div className={styles.div3}>726명 참여중</div>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className={styles.background33}>
                          <img
                            className={styles.logo1jpegIcon2}
                            alt=""
                            src="/logo1jpeg-3@2x.png"
                          />
                          <div className={styles.rectangleContainer}>
                            <div className={styles.separator} />
                          </div>
                          <div className={styles.wrapper}>
                            <div className={styles.div5}>상금 2,100만원</div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.contestDescription}>
                        <div className={styles.samsungAiChallengeBlacParent}>
                          <b className={styles.samsungAiChallenge}>
                            2024 Samsung AI Challenge : Blac…
                          </b>
                          <div className={styles.progress1}>
                            <div className={styles.container21}>
                              <div className={styles.background34} />
                            </div>
                          </div>
                        </div>
                        <div className={styles.container14}>
                          <div className={styles.empty}>2024.08.01</div>
                          <div className={styles.statusSeparator} />
                          <div className={styles.d33}>D-23</div>
                        </div>
                      </div>
                      <div className={styles.container40}>
                        <div className={styles.overlay1}>
                          <div className={styles.div6}>경진대회</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.div6}>알고리즘</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.nlp}>채용</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.nlp}>정형</div>
                        </div>
                        <div className={styles.background6}>
                          <div className={styles.div18}>최적화</div>
                        </div>
                        <div className={styles.background13}>
                          <div className={styles.nrmse}>Recall</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className={styles.cardRow}>
                <div className={styles.link2}>
                  <div className={styles.background54}>
                    <img
                      className={styles.cardCptjpegIcon}
                      alt=""
                      src="/card-cptjpeg-4@2x.png"
                    />
                    <div className={styles.container41}>
                      <div className={styles.container42}>
                        <div className={styles.container43}>
                          <div className={styles.background55}>
                            <div className={styles.strong}>
                              <div className={styles.co}>co</div>
                            </div>
                          </div>
                          <div className={styles.margin5}>
                            <div className={styles.background19}>
                              <img
                                className={styles.picturejpegIcon}
                                alt=""
                                src="/picturejpeg-6@2x.png"
                              />
                            </div>
                          </div>
                          <div className={styles.margin2}>
                            <div className={styles.background19}>
                              <img
                                className={styles.picturejpegIcon}
                                alt=""
                                src="/picturejpeg-7@2x.png"
                              />
                            </div>
                          </div>
                          <div className={styles.marginIcon}>
                            <div className={styles.background19}>
                              <img
                                className={styles.picturejpegIcon}
                                alt=""
                                src="/picturejpeg-8@2x.png"
                              />
                            </div>
                          </div>
                        </div>
                        <div className={styles.margin4}>
                          <div className={styles.overlay}>
                            <div className={styles.div30}>66명 참여중</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.background59}>
                      <div className={styles.logo1jpegContainer}>
                        <img
                          className={styles.logo1jpegIcon4}
                          alt=""
                          src="/logo1jpeg-4@2x.png"
                        />
                      </div>
                      <div className={styles.frameDiv}>
                        <div className={styles.separator} />
                      </div>
                      <div className={styles.div31}>데이스쿨 프로 구독권</div>
                    </div>
                  </div>
                  <div className={styles.contestDescription}>
                    <b className={styles.b2}>
                      식당 리뷰 감성 분석 : 데이스쿨 구독자…
                    </b>
                    <div className={styles.progress1}>
                      <div className={styles.container21}>
                        <div className={styles.background60} />
                      </div>
                    </div>
                    <div className={styles.container14}>
                      <div className={styles.empty}>2024.08.12</div>
                      <div className={styles.statusSeparator} />
                      <div className={styles.d33}>D-19</div>
                    </div>
                  </div>
                  <div className={styles.container47}>
                    <div className={styles.overlay9}>
                      <div className={styles.div18}>해커톤</div>
                    </div>
                    <div className={styles.background61}>
                      <div className={styles.div6}>알고리즘</div>
                    </div>
                    <div className={styles.background61}>
                      <div className={styles.div35}>자연어처리</div>
                    </div>
                    <div className={styles.background61}>
                      <div className={styles.nlp}>분류</div>
                    </div>
                    <div className={styles.background61}>
                      <div className={styles.f1}>F1</div>
                    </div>
                  </div>
                </div>
                <div className={styles.link2}>
                  <div className={styles.background54}>
                    <img
                      className={styles.cardCptjpegIcon}
                      alt=""
                      src="/card-cptjpeg-5@2x.png"
                    />
                    <div className={styles.container41}>
                      <div className={styles.container9}>
                        <div className={styles.container10}>
                          <div className={styles.background1}>
                            <img
                              className={styles.picturejpegIcon}
                              alt=""
                              src="/picturejpeg-9@2x.png"
                            />
                          </div>
                          <div className={styles.margin1}>
                            <div className={styles.background67}>
                              <div className={styles.container11}>
                                <div className={styles.kk}>KK</div>
                              </div>
                            </div>
                          </div>
                          <div className={styles.margin2}>
                            <div className={styles.background19}>
                              <img
                                className={styles.picturejpegIcon}
                                alt=""
                                src="/picturejpeg-10@2x.png"
                              />
                            </div>
                          </div>
                          <img className={styles.marginIcon} />
                        </div>
                        <div className={styles.margin4}>
                          <div className={styles.overlay}>
                            <div className={styles.div3}>500명 참여중</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.background69}>
                      <img
                        className={styles.logo1jpegIcon5}
                        alt=""
                        src="/logo1jpeg-5@2x.png"
                      />
                      <div className={styles.rectangleWrapper1}>
                        <div className={styles.separator} />
                      </div>
                      <div className={styles.div5}>상금 1,700만원</div>
                    </div>
                  </div>
                  <div className={styles.contestDescription}>
                    <b className={styles.samsungAiChallenge}>
                      FSI AIxData Challenge 2024
                    </b>
                    <div className={styles.progress1}>
                      <div className={styles.container21}>
                        <div className={styles.background70} />
                      </div>
                    </div>
                    <div className={styles.container14}>
                      <div className={styles.empty}>2024.08.05</div>
                      <div className={styles.statusSeparator} />
                      <div className={styles.d9}>D-9</div>
                    </div>
                  </div>
                  <div className={styles.container54}>
                    <div className={styles.overlay1}>
                      <div className={styles.div6}>경진대회</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.div6}>알고리즘</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.div35}>금융보안원</div>
                    </div>
                    <div className={styles.background23}>
                      <div className={styles.div8}>생성형 AI</div>
                    </div>
                    <div className={styles.background39}>
                      <div className={styles.nrmse}>생성</div>
                    </div>
                    <div className={styles.background39}>
                      <div className={styles.nrmse}>정형</div>
                    </div>
                    <div className={styles.background39}>
                      <div className={styles.nrmse}>분류</div>
                    </div>
                    <div className={styles.background77}>
                      <div className={styles.oodAuroc}>Macro F1 Score</div>
                    </div>
                    <div className={styles.background78}>
                      <div className={styles.nrmse}>TCAP</div>
                    </div>
                  </div>
                </div>
                <div className={styles.link2}>
                  <div className={styles.background54}>
                    <img
                      className={styles.cardCptjpegIcon}
                      alt=""
                      src="/card-cptjpeg-6@2x.png"
                    />
                    <div className={styles.container41}>
                      <div className={styles.container9}>
                        <div className={styles.container10}>
                          <div className={styles.background80}>
                            <div className={styles.container11}>
                              <div className={styles.mm}>MM</div>
                            </div>
                          </div>
                          <div className={styles.margin5}>
                            <div className={styles.background81}>
                              <div className={styles.strong}>
                                <div className={styles.div1}>재형</div>
                              </div>
                            </div>
                          </div>
                          <div className={styles.margin2}>
                            <div className={styles.background3}>
                              <div className={styles.strong}>
                                <div className={styles.div1}>성우</div>
                              </div>
                            </div>
                          </div>
                          <img className={styles.margin15} />
                        </div>
                        <div className={styles.margin4}>
                          <div className={styles.overlay}>
                            <div className={styles.div3}>987명 참여중</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.background83}>
                      <div className={styles.logo1jpegFrame}>
                        <img
                          className={styles.logo1jpegIcon6}
                          alt=""
                          src="/logo1jpeg-6@2x.png"
                        />
                      </div>
                      <div className={styles.rectangleWrapper}>
                        <div className={styles.separator} />
                      </div>
                      <div className={styles.div5}>상금 1,000만원</div>
                    </div>
                  </div>
                  <div className={styles.contestDescription}>
                    <b className={styles.ai1}>
                      재정정보 AI 검색 알고리즘 경진대회
                    </b>
                    <div className={styles.progress1}>
                      <div className={styles.container61}>
                        <div className={styles.background84} />
                      </div>
                    </div>
                    <div className={styles.container14}>
                      <div className={styles.empty}>2024.07.29</div>
                      <div className={styles.statusSeparator} />
                      <div className={styles.d9}>D-2</div>
                    </div>
                  </div>
                  <div className={styles.container63}>
                    <div className={styles.overlay1}>
                      <div className={styles.div6}>경진대회</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.div6}>알고리즘</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.nlp}>NLP</div>
                    </div>
                    <div className={styles.background23}>
                      <div className={styles.div8}>생성형 AI</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.nlp}>LLM</div>
                    </div>
                    <div className={styles.background14}>
                      <div className={styles.nrmse}>질의응답</div>
                    </div>
                    <div className={styles.background90}>
                      <div className={styles.oodAuroc}>F1 Score</div>
                    </div>
                  </div>
                </div>
                <div className={styles.link2}>
                  <div className={styles.background54}>
                    <img
                      className={styles.cardCptjpegIcon}
                      alt=""
                      src="/card-cptjpeg-7@2x.png"
                    />
                    <div className={styles.container41}>
                      <div className={styles.container9}>
                        <div className={styles.container43}>
                          <div className={styles.background92}>
                            <div className={styles.strong}>
                              <div className={styles.mo}>Mo</div>
                            </div>
                          </div>
                          <div className={styles.margin5}>
                            <div className={styles.background19}>
                              <img
                                className={styles.picturejpegIcon}
                                alt=""
                                src="/picturejpeg-12@2x.png"
                              />
                            </div>
                          </div>
                          <div className={styles.margin2}>
                            <div className={styles.background19}>
                              <img
                                className={styles.picturejpegIcon}
                                alt=""
                                src="/picturejpeg-13@2x.png"
                              />
                            </div>
                          </div>
                          <img className={styles.marginIcon} />
                        </div>
                        <div className={styles.margin4}>
                          <div className={styles.overlay}>
                            <div className={styles.div3}>746명 참여중</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.background95}>
                      <img
                        className={styles.logo1jpegIcon7}
                        alt=""
                        src="/logo1jpeg-7@2x.png"
                      />
                      <div className={styles.rectangleWrapper3}>
                        <div className={styles.separator} />
                      </div>
                      <div className={styles.div55}>상금 4,250만원</div>
                    </div>
                  </div>
                  <div className={styles.contestDescription}>
                    <b className={styles.ai1}>
                      빅데이터 분석 경진대회(분석 실시)
                    </b>
                    <div className={styles.progress7}>
                      <div className={styles.background96} />
                    </div>
                    <div className={styles.container14}>
                      <div className={styles.empty}>2024.06.03</div>
                      <div className={styles.statusSeparator} />
                      <div className={styles.div1}>종료</div>
                    </div>
                  </div>
                  <div className={styles.container69}>
                    <div className={styles.overlay1}>
                      <div className={styles.div6}>경진대회</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.div6}>아이디어</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.div35}>공공데이터</div>
                    </div>
                    <div className={styles.background6}>
                      <div className={styles.div18}>시각화</div>
                    </div>
                    <div className={styles.background100}>
                      <div className={styles.div62}>정성평가</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <footer className={styles.container70} />
            <div className={styles.container71}>
              <div className={styles.heading31}>
                <div className={styles.link6}>
                  <div className={styles.container72}>
                    <b className={styles.b4}>첫걸음 학습 🦶</b>
                  </div>
                  <img
                    className={styles.svgIcon}
                    loading="lazy"
                    alt=""
                    src="/svg.svg"
                  />
                </div>
              </div>
              <div className={styles.div63}>
                위대한 영웅도 초보시절이 있답니다
              </div>
              <div className={styles.courseCarousel}>
                <div className={styles.container73}>
                  <div
                    className={styles.button3}
                    onClick={onButtonContainerClick}
                  >
                    <div className={styles.background}>
                      <img
                        className={styles.summarybannerIcon}
                        loading="lazy"
                        alt=""
                        src="/summarybanner@2x.png"
                      />
                      <div className={styles.container74}>
                        <div className={styles.container75}>
                          <div className={styles.container43}>
                            <div className={styles.background3}>
                              <div className={styles.strong}>
                                <div className={styles.div1}>호준</div>
                              </div>
                            </div>
                            <div className={styles.margin5}>
                              <div className={styles.background103}>
                                <div className={styles.strong}>
                                  <div className={styles.div1}>뽀까</div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin2}>
                              <div className={styles.background104}>
                                <div className={styles.strong}>
                                  <div className={styles.div1}>닉네</div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.marginIcon}>
                              <div className={styles.background3}>
                                <div className={styles.strong}>
                                  <div className={styles.div1}>불타</div>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div className={styles.margin4}>
                            <div className={styles.overlay}>
                              <div className={styles.div3}>784명 참여중</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.courseTitle}>
                      <b className={styles.b5}>판다스 첫걸음 2</b>
                    </div>
                    <div className={styles.courseTags}>
                      <div className={styles.container81}>
                        <div className={styles.background106}>
                          <div className={styles.div18}>판다스</div>
                        </div>
                        <div className={styles.background106}>
                          <div className={styles.div18}>전처리</div>
                        </div>
                        <div className={styles.background106}>
                          <div className={styles.nlp}>분석</div>
                        </div>
                        <div className={styles.background106}>
                          <div className={styles.div72}>데이터프레임</div>
                        </div>
                      </div>
                      <div className={styles.container82}>
                        <div className={styles.background110}>
                          <div className={styles.strong}>
                            <b className={styles.b6}>8</b>
                          </div>
                          <div className={styles.container83}>
                            <div className={styles.stages}>Stages</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className={styles.courseGrid}>
                    <div className={styles.courseRow}>
                      <div className={styles.background}>
                        <img
                          className={styles.summarybannerIcon}
                          alt=""
                          src="/summarybanner@2x.png"
                        />
                        <div className={styles.container74}>
                          <div className={styles.container85}>
                            <div className={styles.container10}>
                              <div className={styles.background1}>
                                <img
                                  className={styles.picturejpegIcon}
                                  alt=""
                                  src="/picturejpeg-14@2x.png"
                                />
                              </div>
                              <div className={styles.margin5}>
                                <div className={styles.background113}>
                                  <div className={styles.strong}>
                                    <div className={styles.div1}>유추</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.margin2}>
                                <div className={styles.background114}>
                                  <div className={styles.strong}>
                                    <div className={styles.div1}>최현</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.margin15}>
                                <div className={styles.background115}>
                                  <div className={styles.container11}>
                                    <div className={styles.an}>JC</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin4}>
                              <div className={styles.overlay}>
                                <div className={styles.div4}>
                                  1,535명 참여중
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.courseTitle}>
                        <b className={styles.b5}>판다스 첫걸음 1</b>
                      </div>
                      <div className={styles.courseTags}>
                        <div className={styles.container90}>
                          <div className={styles.background106}>
                            <div className={styles.div18}>판다스</div>
                          </div>
                          <div className={styles.background106}>
                            <div className={styles.div18}>전처리</div>
                          </div>
                          <div className={styles.background106}>
                            <div className={styles.div18}>시리즈</div>
                          </div>
                        </div>
                        <div className={styles.container82}>
                          <div className={styles.background110}>
                            <div className={styles.strong}>
                              <b className={styles.b6}>7</b>
                            </div>
                            <div className={styles.container83}>
                              <div className={styles.stages}>Stages</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.courseRow}>
                      <div className={styles.background}>
                        <img
                          className={styles.summarybannerIcon}
                          alt=""
                          src="/summarybanner-2@2x.png"
                        />
                        <div className={styles.container74}>
                          <div className={styles.container85}>
                            <div className={styles.container43}>
                              <img className={styles.backgroundIcon} />
                              <div className={styles.margin5}>
                                <div className={styles.background121}>
                                  <div className={styles.strong}>
                                    <div className={styles.div1}>시몬</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.margin2}>
                                <div className={styles.background122}>
                                  <div className={styles.strong}>
                                    <div className={styles.div80}>겨-</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.marginIcon}>
                                <div className={styles.background123}>
                                  <div className={styles.strong}>
                                    <div className={styles.co}>49</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin4}>
                              <div className={styles.overlay}>
                                <div className={styles.div4}>
                                  1,088명 참여중
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.courseTitle}>
                        <b className={styles.b5}>파이썬 첫걸음 2</b>
                      </div>
                      <div className={styles.courseTags}>
                        <div className={styles.container99}>
                          <div className={styles.background106}>
                            <div className={styles.div18}>파이썬</div>
                          </div>
                          <div className={styles.background106}>
                            <div className={styles.nlp}>문법</div>
                          </div>
                        </div>
                        <div className={styles.container82}>
                          <div className={styles.background110}>
                            <div className={styles.strong}>
                              <b className={styles.b6}>6</b>
                            </div>
                            <div className={styles.container83}>
                              <div className={styles.stages}>Stages</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.courseRow}>
                      <div className={styles.background}>
                        <img
                          className={styles.summarybannerIcon}
                          alt=""
                          src="/summarybanner-2@2x.png"
                        />
                        <div className={styles.container74}>
                          <div className={styles.container85}>
                            <div className={styles.container43}>
                              <img className={styles.backgroundIcon1} />
                              <div className={styles.margin5}>
                                <div className={styles.background113}>
                                  <div className={styles.strong}>
                                    <div className={styles.div1}>집에</div>
                                  </div>
                                </div>
                              </div>
                              <div className={styles.margin2}>
                                <div className={styles.background19}>
                                  <img
                                    className={styles.picturejpegIcon}
                                    alt=""
                                    src="/picturejpeg-15@2x.png"
                                  />
                                </div>
                              </div>
                              <div className={styles.marginIcon}>
                                <div className={styles.background130}>
                                  <div className={styles.strong}>
                                    <div className={styles.div1}>도도</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin4}>
                              <div className={styles.overlay}>
                                <div className={styles.div4}>
                                  2,845명 참여중
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.courseTitle}>
                        <b className={styles.b5}>파이썬 첫걸음 1</b>
                      </div>
                      <div className={styles.courseTags}>
                        <div className={styles.container99}>
                          <div className={styles.background106}>
                            <div className={styles.div18}>파이썬</div>
                          </div>
                          <div className={styles.background106}>
                            <div className={styles.nlp}>문법</div>
                          </div>
                        </div>
                        <div className={styles.container82}>
                          <div className={styles.background110}>
                            <div className={styles.strong}>
                              <b className={styles.b6}>7</b>
                            </div>
                            <div className={styles.container83}>
                              <div className={styles.stages}>Stages</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.courseRow}>
                      <div className={styles.background}>
                        <img
                          className={styles.summarybannerIcon}
                          alt=""
                          src="/summarybanner-4@2x.png"
                        />
                        <div className={styles.container74}>
                          <div className={styles.container85}>
                            <div className={styles.container10}>
                              <img className={styles.backgroundIcon2} />
                              <div className={styles.margin1}>
                                <div className={styles.background135}>
                                  <div className={styles.container11}>
                                    <div className={styles.hi}>hi</div>
                                  </div>
                                </div>
                              </div>
                              <img className={styles.margin6} />
                              <div className={styles.margin15}>
                                <div className={styles.background136}>
                                  <div className={styles.container11}>
                                    <div className={styles.div90}>수연</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div className={styles.margin4}>
                              <div className={styles.overlay}>
                                <div className={styles.div91}>
                                  1,738명 참여중
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.wrapper3}>
                        <b className={styles.b13}>넘파이 첫걸음 🔢</b>
                      </div>
                      <div className={styles.courseTags}>
                        <div className={styles.container115}>
                          <div className={styles.background137}>
                            <div className={styles.nrmse}>넘파이</div>
                          </div>
                          <div className={styles.background137}>
                            <div className={styles.nrmse}>Numpy</div>
                          </div>
                        </div>
                        <div className={styles.container82}>
                          <div className={styles.background139}>
                            <div className={styles.strong4}>
                              <b className={styles.b14}>6</b>
                            </div>
                            <div className={styles.container117}>
                              <div className={styles.stages4}>Stages</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className={styles.button4}>
                  <img className={styles.icon1} alt="" src="/icon-11.svg" />
                </div>
                <div className={styles.button5}>
                  <img className={styles.icon1} alt="" src="/icon-21.svg" />
                </div>
              </div>
            </div>
            <div className={styles.button6}>
              <img className={styles.icon3} alt="" src="/icon-3.svg" />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default MiripHome;
