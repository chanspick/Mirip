import PropTypes from "prop-types";
import styles from "./Footer.module.css";

const Footer = ({ className = "" }) => {
  return (
    <footer className={[styles.footer, className].join(" ")}>
      <div className={styles.container}>
        <div className={styles.footerLeft}>
          <div className={styles.container1}>
            <img
              className={styles.footerLogo2e1b5aasvgIcon}
              loading="lazy"
              alt=""
              src="/footerlogo2e1b5aasvg@2x.png"
            />
          </div>
          <div className={styles.footerRight}>
            <div className={styles.container2}>
              <div className={styles.heading6}>
                <div className={styles.ai}>AI 해커톤 플랫폼</div>
              </div>
            </div>
            <div className={styles.container3}>
              <div className={styles.container4}>
                <div className={styles.button}>이용약관</div>
              </div>
              <div className={styles.container5}>
                <div className={styles.link}>
                  <div className={styles.div}>대회 주최 문의</div>
                </div>
              </div>
              <div className={styles.container5}>
                <div className={styles.link}>
                  <div className={styles.div1}>데이콘 서비스 소개</div>
                </div>
              </div>
              <div className={styles.container4}>
                <div className={styles.button1}>개인정보 처리방침</div>
              </div>
              <div className={styles.container5}>
                <div className={styles.link}>
                  <div className={styles.div2}>교육 문의</div>
                </div>
              </div>
              <div className={styles.container5}>
                <div className={styles.link}>
                  <div className={styles.div3}>데이콘 채용</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className={styles.socialLinks}>
          <div className={styles.container10}>
            <div className={styles.link4}>
              <img
                className={styles.kakaoChannel2b21047pngIcon}
                loading="lazy"
                alt=""
                src="/kakao-channel2b21047png@2x.png"
              />
            </div>
            <div className={styles.link5}>
              <img
                className={styles.discord85a3f0dpngIcon}
                loading="lazy"
                alt=""
                src="/discord85a3f0dpng@2x.png"
              />
            </div>
            <div className={styles.link4}>
              <img className={styles.imageIcon} alt="" src="/image-8.svg" />
            </div>
          </div>
          <div className={styles.container11}>
            <div className={styles.container12}>
              <div className={styles.companyInfo}>
                <div className={styles.div4}>
                  <span className={styles.txt}>
                    <p className={styles.p}>
                      데이콘(주) | 대표 김국진 | 699-81-01021
                    </p>
                    <p className={styles.p}>
                      통신판매업 신고번호: 제 2021-서울영등포-1704호
                    </p>
                    <p className={styles.p}>
                      서울특별시 영등포구 은행로 3 익스콘벤처타워 901호
                    </p>
                    <p className={styles.p}>{`이메일 `}</p>
                  </span>
                </div>
                <div className={styles.link7}>
                  <div className={styles.dacondaconio}>dacon@dacon.io</div>
                </div>
                <div className={styles.div5}> | 전화번호: 070-4102-0545</div>
              </div>
              <div className={styles.copyrightDacon}>
                Copyright ⓒ DACON Inc. All rights reserved
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.button2}>
        <img className={styles.icon} alt="" src="/icon-3.svg" />
      </div>
    </footer>
  );
};

Footer.propTypes = {
  className: PropTypes.string,
};

export default Footer;
