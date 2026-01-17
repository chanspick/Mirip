import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import BackgroundHorizontalBorder from "../components/BackgroundHorizontalBorder";
import styles from "./MiripEdu.module.css";

const MiripEdu = () => {
  const navigate = useNavigate();

  const onLinkContainerClick = useCallback(() => {
    navigate("/");
  }, [navigate]);

  return (
    <div className={styles.miripEdu}>
      <main className={styles.container}>
        <BackgroundHorizontalBorder
          mainLogob9ffbb6svg="/mainlogob9ffbb6svg1.svg"
          linkWidth="128px"
          aColor="#5d6970"
          containerColor="#3c7cde"
          buttonWidth="385px"
          onLinkContainerClick={onLinkContainerClick}
          linkBoxShadow="unset"
          linkBackgroundColor="unset"
          linkBorder="unset"
        />
        <section className={styles.main}>
          <div className={styles.container1}>
            <div className={styles.container2}>
              <div className={styles.background}>
                <h3 className={styles.h3}>판다스 첫걸음 2</h3>
                <div className={styles.div}>
                  전처리, 분석, 데이터프레임, 판다스
                </div>
                <div className={styles.item}>
                  <img
                    className={styles.icon}
                    loading="lazy"
                    alt=""
                    src="/icon2.svg"
                  />
                  <div className={styles.div1}>첫걸음 프로젝트</div>
                </div>
                <div className={styles.item1}>
                  <img className={styles.icon} alt="" src="/icon-12.svg" />
                  <div className={styles.div2}>5 시간</div>
                  <div className={styles.margin}>
                    <div className={styles.container3}>
                      <div className={styles.container4}>
                        <div className={styles.container5}>
                          <img
                            className={styles.icon2}
                            alt=""
                            src="/icon-22.svg"
                          />
                        </div>
                      </div>
                      <div className={styles.div3}>8 스테이지</div>
                    </div>
                  </div>
                </div>
                <div className={styles.item2}>
                  <div className={styles.iconWrapper}>
                    <img className={styles.icon2} alt="" src="/icon-1.svg" />
                  </div>
                  <div className={styles.div4}>787 명</div>
                </div>
              </div>
              <div className={styles.horizontalborder}>
                <div className={styles.container6}>
                  <div className={styles.button}>
                    <div className={styles.wrapper}>
                      <b className={styles.b}>프로젝트</b>
                    </div>
                    <div className={styles.horizontalDivider} />
                  </div>
                </div>
              </div>
              <div className={styles.container7}>
                <div className={styles.container8}>
                  <div className={styles.container9}>
                    <div className={styles.heading3}>
                      <h3 className={styles.h31}>프로젝트 설명</h3>
                    </div>
                    <div className={styles.container10}>
                      <div className={styles.div5}>
                        어떤 프로젝트일지 시작하기 전에 읽어보세요
                      </div>
                    </div>
                    <div className={styles.background1}>
                      <div className={styles.container11}>
                        <div className={styles.heading3}>
                          <b className={styles.strong}>개요</b>
                        </div>
                        <div className={styles.heading3}>
                          <div className={styles.pythonContainer}>
                            <p className={styles.python}>
                              이 교재의 목적은 데이터 분석을 위한 Python
                              라이브러리 중 하나인 판다스(Pandas)와 데이터프레임
                            </p>
                            <p className={styles.python}>
                              (DataFrame)에 대한 포괄적인 학습 자료를 제공하는
                              것입니다. 이를 통해 독자들은 데이터를 효과적으로
                              다
                            </p>
                            <p className={styles.python}>
                              루고 분석할 수 있는 능력을 키울 것입니다.
                            </p>
                          </div>
                        </div>
                        <div className={styles.heading3}>
                          <b className={styles.strong}>목표</b>
                        </div>
                        <div className={styles.heading3}>
                          <div className={styles.div6}>
                            <p className={styles.python}>
                              데이터프레임은 데이터를 표 형태로 다루는 판다스의
                              핵심 자료구조입니다. 이를 이해하고 활용하기 위해
                              데
                            </p>
                            <p className={styles.python}>
                              이터프레임의 생성, 읽기, 갱신, 삭제에 대한 방법을
                              학습하게 됩니다. 이러한 과정을 통해 데이터를
                              효과적으
                            </p>
                            <p className={styles.python}>
                              로 다루고 관리할 수 있는 능력을 기를 수 있습니다.
                            </p>
                          </div>
                        </div>
                        <div className={styles.heading3}>
                          <b className={styles.strong}>설명</b>
                        </div>
                        <div className={styles.heading3}>
                          <div className={styles.div7}>
                            <span>
                              <p className={styles.python}>
                                판다스는 비정형데이터 처리의 기본이 되는
                                라이브러리 입니다. 판다스는 데이터프레임과
                                시리즈를 통해 표
                              </p>
                              <p className={styles.python}>
                                형태의 데이터를 다루기 쉽게 만들어주며, 데이터의
                                필터링, 변환, 결합 등 다양한 연산을 수행할 수
                                있습니다.
                              </p>
                              <p className={styles.python}>
                                이를 통해 데이터 과학자들과 엔지니어들은 복잡한
                                데이터를 효과적으로 처리하고 분석할 수 있습니다.
                                따라
                              </p>
                              <p className={styles.python}>
                                서 판다스는 데이터 분석 및 가공 작업을 위한
                                필수적인 도구로 자리매김하고 있습니다.
                              </p>
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className={styles.parent}>
                    <h3 className={styles.h31}>프로젝트 과정</h3>
                    <div className={styles.div8}>
                      차근차근 단계를 밟아 학습해보세요.
                    </div>
                    <div className={styles.horizontalborder1}>
                      <input
                        className={styles.input}
                        placeholder="스테이지 8 개"
                        type="text"
                      />
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container19}
                            placeholder="1. 데이터프레임 생성하기"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container21}
                            placeholder="2. 데이터프레임 속성과 메소드"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container23}
                            placeholder="3. 데이터프레임의 인덱싱과 슬라이싱"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container25}
                            placeholder="4. 데이터프레임의 데이터 필터링"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container27}
                            placeholder="5. 데이터프레임 열 조작과 정렬"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container29}
                            placeholder="6. 데이터 조작 심화"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container31}
                            placeholder="7. 데이터프레임 결합"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                    <div className={styles.container18}>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.background2} />
                      </div>
                      <div className={styles.margin1}>
                        <div className={styles.backgroundborder}>
                          <input
                            className={styles.container33}
                            placeholder="8. 결측치 처리와 중복 제거, 피벗테이블"
                            type="text"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div className={styles.margin9}>
                  <div className={styles.container34}>
                    <div className={styles.border}>
                      <div className={styles.heading3}>
                        <b className={styles.b1}>내 학습 진도</b>
                      </div>
                      <div className={styles.margin10}>
                        <div className={styles.container35}>
                          <div className={styles.backgroundborder8}>
                            <div className={styles.container5}>
                              <div className={styles.div9}>
                                1. 데이터프레임 생성하기
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className={styles.buttonmargin}>
                        <button className={styles.button1}>
                          <div className={styles.div10}>이어서 학습하기</div>
                        </button>
                      </div>
                    </div>
                    <div className={styles.section}>
                      <div className={styles.background10}>
                        <img
                          className={styles.picturejpegIcon}
                          loading="lazy"
                          alt=""
                          src="/picturejpeg1@2x.png"
                        />
                      </div>
                      <div className={styles.backgroundWrapper}>
                        <div className={styles.heading41}>
                          <b className={styles.b2}>데이스쿨</b>
                        </div>
                      </div>
                      <div className={styles.margin11}>
                        <div className={styles.paragraph}>
                          <div className={styles.div11}>
                            <span>
                              <p className={styles.python}>안녕하세요!🙋</p>
                              <p className={styles.python}>
                                데이스쿨은 인공지능 초/중급 학습자를 위한
                                프로젝트 학
                              </p>
                              <p className={styles.python}>
                                습📚, 스터디👥, 해커톤🖥️으로 구성된 학습
                                플랫폼입니다.
                              </p>
                              <p className={styles.python}>
                                데이스쿨은 여러분이 인공지능 분야에서 실력을
                                쌓고, 성
                              </p>
                              <p className={styles.python}>
                                장하는 데 필요한 학습을 제공합니다.🌟
                              </p>
                              <p className={styles.python}>
                                부단한 연습과 노력을 통해 여러분의 학습 목표를
                                달성해
                              </p>
                              <p className={styles.python}>보세요.💪</p>
                              <p className={styles.python}>
                                매일의 작은 노력이 모여 큰 성공으로 이어집니다.
                                🏆
                              </p>
                              <p className={styles.python}>
                                여러분의 성공을 위해 데이스쿨이 함께 하겠습니다.
                                🎉
                              </p>
                            </span>
                          </div>
                          <div className={styles.dacon0schoolgmailcom}>
                            📧 연락처: dacon0school@gmail.com
                          </div>
                        </div>
                      </div>
                      <div className={styles.buttonmargin1}>
                        <button className={styles.button2}>
                          <div className={styles.div12}>더보기</div>
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default MiripEdu;
