import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import BackgroundHorizontalBorder from "../components/BackgroundHorizontalBorder";
import Footer from "../components/Footer";
import styles from "./MiripComp.module.css";

const MiripComp = () => {
  const navigate = useNavigate();

  const onLinkContainerClick = useCallback(() => {
    navigate("/");
  }, [navigate]);

  return (
    <div className={styles.miripComp}>
      <main className={styles.container}>
        <BackgroundHorizontalBorder
          mainLogob9ffbb6svg="/mainlogob9ffbb6svg.svg"
          onLinkContainerClick={onLinkContainerClick}
        />
        <section className={styles.main}>
          <div className={styles.container1}>
            <div className={styles.background}>
              <b className={styles.nh}>2024 NH 투자증권 빅데이터 경진대회</b>
              <div className={styles.nh1}>
                빅데이터 | NH | 분석 | AI | ETF | Microsoft | Tableau | 시각화 |
                아이디어
              </div>
              <div className={styles.container2}>
                <div className={styles.list}>
                  <div className={styles.item}>
                    <img
                      className={styles.imageIcon}
                      alt=""
                      src="/image-1.svg"
                    />
                    <div className={styles.div}> 상금 : 6,300만원</div>
                  </div>
                  <div className={styles.frameParent}>
                    <div className={styles.frameGroup}>
                      <div className={styles.iconWrapper}>
                        <img
                          className={styles.icon}
                          loading="lazy"
                          alt=""
                          src="/icon.svg"
                        />
                      </div>
                      <div className={styles.wrapper}>
                        <div
                          className={styles.div1}
                        >{` 2024.09.02 ~ 2024.10.11 08:00 `}</div>
                      </div>
                      <div className={styles.linkWrapper}>
                        <div className={styles.link}>
                          <a className={styles.googleCalendar}>
                            + Google Calendar
                          </a>
                        </div>
                      </div>
                    </div>
                    <div className={styles.frameContainer}>
                      <div className={styles.frameDiv}>
                        <div className={styles.iconContainer}>
                          <img
                            className={styles.icon}
                            alt=""
                            src="/icon-1.svg"
                          />
                        </div>
                        <div className={styles.div2}> 234명 </div>
                      </div>
                      <div className={styles.frameWrapper}>
                        <div className={styles.frameParent1}>
                          <div className={styles.iconFrame}>
                            <img
                              className={styles.icon}
                              alt=""
                              src="/icon-2.svg"
                            />
                          </div>
                          <div className={styles.d50}> D-50</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className={styles.container3}>
                  <img
                    className={styles.imageIcon1}
                    alt=""
                    src="/image-2.svg"
                  />
                  <div className={styles.link1}>
                    <b className={styles.b}>참여</b>
                  </div>
                </div>
              </div>
            </div>
            <div className={styles.container4}>
              <div className={styles.tablist}>
                <div className={styles.tabParent}>
                  <div className={styles.tab}>
                    <div className={styles.frame}>
                      <b className={styles.b1}>대회안내</b>
                    </div>
                    <div className={styles.container5}>
                      <div className={styles.horizontalDivider} />
                    </div>
                  </div>
                  <div className={styles.tab1}>
                    <div className={styles.div3}>데이터</div>
                  </div>
                  <div className={styles.tab2}>
                    <div className={styles.div4}>코드 공유</div>
                  </div>
                  <div className={styles.tab3}>
                    <div className={styles.div5}>토크</div>
                  </div>
                  <div className={styles.tab4}>
                    <div className={styles.div6}>리더보드</div>
                  </div>
                  <div className={styles.tab5}>
                    <div className={styles.div7}>팀</div>
                  </div>
                  <div className={styles.tab3}>
                    <div className={styles.div5}>제출</div>
                  </div>
                </div>
                <div className={styles.separator} />
              </div>
              <div className={styles.container6}>
                <div className={styles.backgroundborder}>
                  <div className={styles.background1}>
                    <button className={styles.item1}>
                      <div className={styles.imgmarginParent}>
                        <div className={styles.imgmargin}>
                          <img
                            className={styles.imageIcon2}
                            alt=""
                            src="/image-3.svg"
                          />
                        </div>
                        <div className={styles.verticalDivider} />
                      </div>
                      <div className={styles.overviewItem}>
                        <a
                          className={styles.a}
                          href="https://www.figma.com/design/6j4gjhd0s2QvCx2By6rIpM?node-id=7-2"
                          target="_blank"
                        >
                          개요
                        </a>
                      </div>
                    </button>
                    <div className={styles.item2}>
                      <div className={styles.link2}>
                        <div className={styles.imgmargin1}>
                          <img
                            className={styles.imageIcon2}
                            alt=""
                            src="/image-4.svg"
                          />
                        </div>
                        <div className={styles.container7}>
                          <div className={styles.div9}>규칙</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.item2}>
                      <div className={styles.link2}>
                        <div className={styles.imgmargin1}>
                          <img
                            className={styles.imageIcon2}
                            alt=""
                            src="/image-5.svg"
                          />
                        </div>
                        <div className={styles.container7}>
                          <div className={styles.div9}>일정</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.item2}>
                      <div className={styles.link2}>
                        <div className={styles.imgmargin1}>
                          <img
                            className={styles.imageIcon2}
                            alt=""
                            src="/image-6.svg"
                          />
                        </div>
                        <div className={styles.container7}>
                          <div className={styles.div9}>상금</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.item2}>
                      <div className={styles.link5}>
                        <div className={styles.imgmargin1}>
                          <img
                            className={styles.imageIcon2}
                            alt=""
                            src="/image-7.svg"
                          />
                        </div>
                        <div className={styles.container7}>
                          <div className={styles.div12}>동의사항</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className={styles.backgroundverticalborder}>
                    <div className={styles.heading3} />
                    <div className={styles.flzqcxy3n4djde4shafuqx89djpgParent}>
                      <img
                        className={styles.flzqcxy3n4djde4shafuqx89djpgIcon}
                        loading="lazy"
                        alt=""
                        src="/flzqcxy3n4djde4shafuqx89djpg@2x.png"
                      />
                      <div className={styles.parent}>
                        <b className={styles.b2}>[배경]</b>
                        <div className={styles.aiEtf}>
                          2024 NH투자증권 빅데이터 경진대회에 오신 것을
                          환영합니다.
                        </div>
                        <div className={styles.aiEtf}>
                          AI로 미국 ETF를 발견할 수 있는 기회!
                        </div>
                        <div className={styles.aiEtf}>
                          인공지능으로 해외 ETF의 투자 잠재력을 발견하고, 멋진
                          추억 만들기! 지금 대회에 도전하세요.
                        </div>
                        <div className={styles.aiEtf}>
                          5회째를 맞이하는 NH투자증권 빅데이터 경진대회,
                        </div>
                        <div className={styles.aiEtf}>
                          올해는 글로벌 테크기업 마이크로소프트, 태블로와 함께
                          합니다.
                        </div>
                        <div className={styles.aiEtf}>
                          2024 NH투자증권 빅데이터 경진대회에 참여하고,
                        </div>
                        <div className={styles.aiEtf}>
                          최대 상금 1,500만원과 다양한 경품으로 멋진 기억을
                          남겨보세요.
                        </div>
                      </div>
                    </div>
                    <div className={styles.container11} />
                    <div className={styles.group}>
                      <b className={styles.b3}>[주제]</b>
                      <div className={styles.ai}>
                        생성형 AI를 활용한 미국 ETF 큐레이션 서비스 제안
                      </div>
                    </div>
                    <div className={styles.container11} />
                    <div className={styles.parent1}>
                      <b className={styles.b4}>[주최 / 후원 / 주관]</b>
                      <div className={styles.list1}>
                        <div className={styles.item6}>
                          <div className={styles.div15}>•</div>
                          <div className={styles.nh5}>주최: NH투자증권</div>
                        </div>
                        <div className={styles.item6}>
                          <div className={styles.div15}>•</div>
                          <div className={styles.div17}>
                            후원: 마이크로소프트, 태블로
                          </div>
                        </div>
                        <div className={styles.item6}>
                          <div className={styles.div15}>•</div>
                          <div className={styles.div19}>주관: 데이콘</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.container11} />
                    <div className={styles.group}>
                      <b className={styles.b4}>[참가 대상]</b>
                      <div className={styles.div20}>
                        대회 참여일 기준 국내/해외 대학(원)생 개인 및 최대 3인
                        구성 팀
                      </div>
                    </div>
                    <div className={styles.container11} />
                    <div className={styles.parent3}>
                      <div className={styles.aiEtf}>
                        ※ 대학(원) 재학 중인 직장 근로소득자 및 사업등록자는
                        참가 대상에서 제외
                      </div>
                      <div className={styles.aiEtf}>
                        ※ 대학(원) 재학생 범위는 재학생, 휴학생, 졸업 유예생으로
                        증명서 발급이 가능한 자로 한정
                      </div>
                    </div>
                  </div>
                </div>
                <div className={styles.border}>
                  <div className={styles.heading3Wrapper}>
                    <b className={styles.heading31}>대회 주요 일정</b>
                  </div>
                  <div className={styles.scheduleBlocksParent}>
                    <div className={styles.scheduleBlocks}>
                      <div className={styles.scheduleBackgrounds}>
                        <div className={styles.background2} />
                        <div className={styles.background3}>
                          <img
                            className={styles.imageFillIcon}
                            loading="lazy"
                            alt=""
                            src="/image-fill@2x.png"
                          />
                          <div className={styles.emptyCell}>08.12</div>
                          <div className={styles.div24}>참가 신청 시작</div>
                        </div>
                      </div>
                      <div className={styles.scheduleBackgrounds}>
                        <div className={styles.background2} />
                        <div className={styles.background5}>
                          <div className={styles.emptyCell}>11.17</div>
                          <div className={styles.div25}>본선 마감</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.scheduleBlocks}>
                      <div className={styles.scheduleBackgrounds}>
                        <div className={styles.background2} />
                        <div className={styles.background5}>
                          <div className={styles.emptyCell}>09.02</div>
                          <div className={styles.div27}>데이터 오픈</div>
                        </div>
                      </div>
                      <div className={styles.scheduleBackgrounds}>
                        <div className={styles.background8} />
                        <div className={styles.background9}>
                          <div className={styles.emptyCell}>11.22</div>
                          <div className={styles.div25}>결선 시작</div>
                        </div>
                      </div>
                    </div>
                    <div className={styles.scheduleBlocks}>
                      <div className={styles.scheduleBackgrounds}>
                        <div className={styles.background2} />
                        <div className={styles.background5}>
                          <div className={styles.emptyCell}>10.11</div>
                          <div className={styles.div31}>
                            <span>
                              <p className={styles.p}>과제물 제출 및 팀</p>
                              <p className={styles.p}>병합 마감(예선 종</p>
                              <p className={styles.p}>료)</p>
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className={styles.item9}>
                        <div className={styles.emptyCell}>11.29</div>
                        <div className={styles.div33}>
                          <span>
                            <p className={styles.p}>결선 발표 및 시상</p>
                            <p className={styles.p}>(쇼케이스)</p>
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className={styles.scheduleBlocks3}>
                      <div className={styles.background2} />
                      <div className={styles.background5}>
                        <div className={styles.emptyCell}>10.17</div>
                        <div className={styles.div24}>예선 평가 마감</div>
                      </div>
                    </div>
                    <div className={styles.item10}>
                      <div className={styles.emptyCell}>10.18</div>
                      <div className={styles.div25}>본선 시작</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <Footer />
        </section>
      </main>
    </div>
  );
};

export default MiripComp;
